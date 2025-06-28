#!/usr/bin/env python
import pathlib, sys
# ------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]  # …/src
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------
"""
generate_dummy_outcomes.py
──────────────────────────
Populate the local SQLite DB with:

  • N synthetic tasks
  • 1…3 synthetic assignments per task
  • A matching AssignmentOutcome row for every assignment

Optionally POST those outcomes to the backend so the
learning stack (feedback → analytics) runs just like production.

USAGE
=====

# populate DB only
python src/scripts/generate_dummy_outcomes.py --tasks 50

# populate DB *and* call backend
python src/scripts/generate_dummy_outcomes.py \
        --tasks 50 \
        --post http://localhost:8001/api/v1/learning/feedback
"""
import random, argparse, os, sys, json, datetime, requests
from typing import List
from sqlalchemy.orm import Session
from models.database import (
    SessionLocal, Developer, Task, TaskAssignment, AssignmentOutcome
)

# ────────────────────────────────────────────────────────────────────
# Helper generators
# ────────────────────────────────────────────────────────────────────
TECH_SKILLS  = ["python","typescript","cpp","go","java","rust"]
DOMAINS      = ["ml","backend","frontend","devops","security","testing"]

def make_task(i: int) -> Task:
    return Task(
        title       = f"Dummy Feature #{i}",
        description = f"Implement feature #{i} related to {random.choice(DOMAINS)}",
        repository  = "dummy/repo",
        labels      = ["enhancement"],
        priority    = random.choice(["low","medium","high"]),
        technical_complexity   = round(random.uniform(0.2,0.9),2),
        domain_difficulty      = round(random.uniform(0.2,0.9),2),
        collaboration_requirements = round(random.uniform(0.1,0.9),2),
        learning_opportunities = round(random.uniform(0.2,0.9),2),
        business_impact        = round(random.uniform(0.2,0.9),2),
        estimated_hours        = round(random.uniform(4,40),1),
        complexity_confidence  = round(random.uniform(0.6,0.9),2),
        required_skills        = {random.choice(TECH_SKILLS): round(random.uniform(0.3,0.9),2)}
    )

def make_outcome(assignment: TaskAssignment) -> AssignmentOutcome:
    # “Ground-truth” inversely correlated with complexity
    c = (assignment.task.technical_complexity or 0.5)
    quality = max(0.1, 1.0 - c + random.uniform(-0.1,0.1))
    return AssignmentOutcome(
        assignment_id            = assignment.id,
        task_completion_quality  = round(quality,2),
        developer_satisfaction   = round(random.uniform(0.4,1.0),2),
        learning_achieved        = round(random.uniform(0.2,0.9),2),
        collaboration_effectiveness = round(random.uniform(0.3,1.0),2),
        time_estimation_accuracy = round(random.uniform(0.4,0.9),2),
        performance_metrics      = {},
        skill_improvements       = [],
        challenges_faced         = [],
        success_factors          = [],
    )

# ────────────────────────────────────────────────────────────────────
def seed_db(num_tasks: int, session: Session) -> List[AssignmentOutcome]:
    devs = session.query(Developer).all()
    if not devs:
        raise RuntimeError("➡ You have 0 developers in the DB – run /api/v1/developers first.")

    outcomes: List[AssignmentOutcome] = []

    for i in range(num_tasks):
        task = make_task(i)
        session.add(task)
        session.flush()                       # get task.id

        # 1…3 assignments
        assignees = random.sample(devs, min(len(devs), random.randint(1,3)))
        for dev in assignees:
            assign = TaskAssignment(
                task_id        = task.id,
                developer_id   = dev.id,
                status         = "completed",
                confidence_score = round(random.uniform(0.4,0.9),2),
                reasoning      = "synthetic seed",
                assigned_at    = datetime.datetime.utcnow() - datetime.timedelta(days=random.randint(1,30)),
                completed_at   = datetime.datetime.utcnow(),
                actual_hours   = task.estimated_hours * random.uniform(0.8,1.3),
            )
            session.add(assign)
            session.flush()                   # get assignment.id

            out = make_outcome(assign)
            outcomes.append(out)
            session.add(out)

    session.commit()
    return outcomes

# ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=30, help="How many tasks to create")
    ap.add_argument("--post",  help="POST JSON payload to this /learning/feedback URL")
    args = ap.parse_args()

    db: Session = SessionLocal()
    outcomes = seed_db(args.tasks, db)
    print(f"✅  Inserted {len(outcomes)} outcomes into DB.")

    if args.post:
        payload = [ {
            "assignment_id": o.assignment_id,
            "task_completion_quality":   o.task_completion_quality,
            "developer_satisfaction":    o.developer_satisfaction,
            "learning_achieved":         o.learning_achieved,
            "collaboration_effectiveness": o.collaboration_effectiveness,
            "time_estimation_accuracy":  o.time_estimation_accuracy,
            "performance_metrics": {},
            "skill_improvements": [],
            "challenges_faced": [],
            "success_factors": []
        } for o in outcomes ]

        r = requests.post(args.post, json=payload, timeout=60)
        if r.ok:
            print(f"🌐  Posted {len(payload)} outcomes → {args.post} | status {r.status_code}")
        else:
            print(f"⚠️  POST failed – {r.status_code}: {r.text[:200]}")

if __name__ == "__main__":
    main()
