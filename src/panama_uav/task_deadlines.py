"""Read-only sensing-to-robot-command deadline accounting in seconds."""

DEADLINE_TOLERANCE_SECONDS = 1e-12


def command_deadline_fields(task, observed_at):
    """Classify one logical task using its first complete command arrival.

    Undelivered latency is right-censored. Once the observation horizon reaches
    the inclusive deadline, absence of an arrival is a known deadline miss even
    though final latency remains unknown. No queue or execution state is changed.
    """
    arrival, bound = task.get("delivered_at"), task.get("deadline_at")
    delivered, enabled = arrival is not None, bound is not None
    if delivered and arrival > observed_at+DEADLINE_TOLERANCE_SECONDS:
        raise ValueError("Cannot measure a future command arrival")
    on_time = delivered and enabled and arrival <= bound+DEADLINE_TOLERANCE_SECONDS
    overdue = enabled and not delivered and observed_at >= bound
    missed = enabled and ((delivered and not on_time) or overdue)
    status = ("on_time" if on_time else "late" if delivered else "undelivered") if enabled else "disabled"
    return dict(task_deadline_enabled=enabled, deadline_status=status,
        deadline_met=True if on_time else False if missed else None,
        task_deadline_missed=bool(missed), deadline_censored=enabled and not delivered,
        deadline_pending=enabled and not delivered and not overdue,
        deadline_undelivered_overdue=bool(overdue), deadline_observed_at=observed_at,
        command_delivery_status="delivered" if delivered else "undelivered",
        deadline_latency_lower_bound_seconds=observed_at-task["source_time"] if not delivered else None)


def command_deadline_metrics(tasks, observed_at):
    """Exclusive outcome counts, with WAIT/censor tags kept out of miss totals."""
    ids = [task["job_id"] for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("Deadline accounting requires one record per logical task")
    monitored = [{**task, **command_deadline_fields(task, observed_at)} for task in tasks
                 if task.get("deadline_at") is not None]
    on_time = sum(t["deadline_status"] == "on_time" for t in monitored)
    late = sum(t["deadline_status"] == "late" for t in monitored)
    undelivered = sum(t["deadline_status"] == "undelivered" for t in monitored)
    overdue = sum(t["deadline_undelivered_overdue"] for t in monitored)
    pending = sum(t["deadline_pending"] for t in monitored)
    assessed = on_time+late+overdue
    return dict(deadline_tasks_monitored=len(monitored), deadline_tasks_assessed=assessed,
        deadline_commands_on_time=on_time if monitored else None,
        deadline_commands_late=late if monitored else None,
        deadline_commands_undelivered=undelivered if monitored else None,
        deadline_undelivered_overdue_tasks=overdue, deadline_pending_tasks=pending,
        task_deadline_misses=late+overdue, deadline_censored_tasks=undelivered,
        deadline_cutoff_wait_tasks=sum(t.get("cutoff_induced_wait") is True for t in monitored),
        deadline_compliance_rate=on_time/assessed if assessed else None)
