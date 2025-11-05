"""Deploy Prefect flow to cloud."""

from prefect import flow
from prefect_flow import hospital_readmission_pipeline
from prefect.deployments import Deployment
from prefect.schedules import IntervalSchedule
from datetime import timedelta

# Create deployment with daily schedule
deployment = Deployment.build(
    flow=hospital_readmission_pipeline,
    name="hospital-readmission-daily",
    description="Daily hospital readmission prediction pipeline",
    schedule=IntervalSchedule(interval=timedelta(days=1)),
    work_queue_name="default",
    tags=["production", "daily"],
    parameters={}
)

if __name__ == "__main__":
    deployment.apply()
    print("✓ Deployment created successfully!")
    print("Run with: prefect deployment run 'hospital-readmission-daily/hospital-readmission-daily'")