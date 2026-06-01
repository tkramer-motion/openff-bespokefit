import asyncio
import json
import logging
import time
import traceback
from collections import deque

# Fix for OpenEye segfaults in forked processes
try:
    from openeye import oechem
    if oechem.OEChemIsLicensed() and oechem.OEGetMemPoolMode() == oechem.OEMemPoolMode_Default:
        oechem.OESetMemPoolMode(oechem.OEMemPoolMode_Mutexed | oechem.OEMemPoolMode_UnboundedCache)
except (ImportError, ModuleNotFoundError):
    pass

import redis

from openff.bespokefit.executor.services import current_settings
from openff.bespokefit.executor.services.coordinator.storage import (
    TaskStatus,
    get_n_tasks,
    get_task,
    pop_task_status,
    push_task_status,
    save_task,
)

_logger = logging.getLogger(__name__)

# Debug info storage
_debug_info = {
    "cycle_count": 0,
    "last_cycle_time": None,
    "last_error": None,
    "recent_errors": deque(maxlen=10),
    "tasks_processed": 0,
}


async def _process_task(task_id: int) -> bool:
    task = get_task(task_id)
    task_status = task.status

    if task.status == "success" or task.status == "errored":
        return True

    try:
        if task.running_stage is None:
            task.running_stage = task.pending_stages.pop(0)
            await task.running_stage.enter(task)

        stage_status = task.running_stage.status
        await task.running_stage.update()

        task_state_message = f"[task id={task_id}] transitioned from {{0}} -> {{1}}"

        if task.status != task_status and task_status == "waiting":
            print(task_state_message.format(task_status, task.status), flush=True)

        if stage_status != task.running_stage.status:
            print(
                f"[task id={task_id}] {task.running_stage.type} transitioned from "
                f"{stage_status} -> {task.running_stage.status}",
                flush=True,
            )

        if task.running_stage.status in {"success", "errored"}:
            task.completed_stages.append(task.running_stage)
            task.running_stage = None

        if task.status != task_status and task_status != "waiting":
            print(task_state_message.format(task_status, task.status), flush=True)
    except Exception as e:  # noqa: BLE001 - surface, don't silently re-process
        # If processing this task raises, the cycle's broad handler would otherwise
        # swallow it and re-enter this task next cycle without advancing it - which
        # manifests as repeated stage transitions and an eventual silent drop. Instead
        # fail the task once, loudly, with the real error so the client can report it.
        _logger.exception("error while processing task %s", task_id)
        if task.running_stage is None and task.pending_stages:
            task.running_stage = task.pending_stages.pop(0)
        if task.running_stage is not None:
            task.running_stage.status = "errored"
            task.running_stage.error = json.dumps(f"{type(e).__name__}: {str(e)}")
            task.completed_stages.append(task.running_stage)
            task.running_stage = None

    save_task(task)
    return task.status in ("success", "errored")


async def cycle():  # pragma: no cover
    settings = current_settings()
    n_connection_errors = 0
    n_consecutive_errors = 0

    while True:
        sleep_time = settings.BEFLOW_COORDINATOR_MAX_UPDATE_INTERVAL

        try:
            start_time = time.perf_counter()
            _debug_info["cycle_count"] += 1
            _debug_info["last_cycle_time"] = time.time()

            # First update any running tasks, pushing them to the 'complete' queue if
            # they have finished, so as to figure out how many new tasks can be moved
            # from running to waiting.

            # Claim each running task by POPPING it, process THAT task, then move it on
            # based on ITS OWN result. This must not be a peek-then-pop split: peeking the
            # head, processing it, then popping "the head" again can move a *different*
            # task than the one processed (the head may change across the awaits), which
            # applies one task's `has_finished` verdict to another -- pushing an
            # unfinished task to 'complete' and orphaning it (its stage stuck 'running'
            # forever while the client's wait_until_complete hangs indefinitely).
            processed_task_ids = set()
            task_id = pop_task_status(TaskStatus.running)

            while task_id is not None:
                if task_id in processed_task_ids:
                    # We've cycled through every task currently in the running queue
                    # (this one was already handled and re-queued as unfinished). Put it
                    # back untouched and stop for this cycle.
                    push_task_status(task_id, TaskStatus.running)
                    break

                try:
                    has_finished = await _process_task(task_id)
                    _debug_info["tasks_processed"] += 1
                    # Needed to let other async threads run even if there are hundreds of
                    # tasks running
                    await asyncio.sleep(0.0)
                except BaseException:
                    # The task is currently claimed (out of every queue); never leave it
                    # orphaned on error -- put it back on 'running' so it is retried.
                    push_task_status(task_id, TaskStatus.running)
                    raise

                push_task_status(
                    task_id,
                    TaskStatus.complete if has_finished else TaskStatus.running,
                )
                processed_task_ids.add(task_id)

                task_id = pop_task_status(TaskStatus.running)

            n_running_tasks = get_n_tasks(TaskStatus.running)
            n_tasks_to_queue = min(
                settings.BEFLOW_COORDINATOR_MAX_RUNNING_TASKS - n_running_tasks,
                get_n_tasks(TaskStatus.waiting),
            )

            for _ in range(n_tasks_to_queue):
                waiting_task_id = pop_task_status(TaskStatus.waiting)
                if waiting_task_id is not None:
                    push_task_status(waiting_task_id, TaskStatus.running)

            n_connection_errors = 0
            n_consecutive_errors = 0

            # Make sure we don't cycle too often
            sleep_time = max(sleep_time - (time.perf_counter() - start_time), 0.0)

        except (KeyboardInterrupt, asyncio.CancelledError):
            break

        except (
            ConnectionError,
            redis.exceptions.ConnectionError,
            redis.exceptions.BusyLoadingError,
        ) as e:
            n_connection_errors += 1
            error_info = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc(), "time": time.time()}
            _debug_info["last_error"] = error_info
            _debug_info["recent_errors"].append(error_info)

            if n_connection_errors >= 3:
                raise e

            if isinstance(e, redis.exceptions.RedisError):
                _logger.warning(
                    f"Failed to connect to Redis - {3 - n_connection_errors} attempts "
                    f"remaining."
                )

        except Exception as e:
            n_consecutive_errors += 1
            error_info = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc(), "time": time.time()}
            _debug_info["last_error"] = error_info
            _debug_info["recent_errors"].append(error_info)
            print(f"Coordinator error: {error_info}", flush=True)

            # Don't spin forever on a persistent error. Give up so the failure surfaces
            # (the client health-check turns a stopped coordinator into a clear error
            # for the caller instead of an indefinite hang).
            if n_consecutive_errors >= 10:
                raise

        await asyncio.sleep(sleep_time)
