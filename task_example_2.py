from clearml import Task
import pandas as pd

task = Task.init(project_name="test_project", task_name='task_artifacts_example')

for i in range(25):
    task.get_logger().report_scalar("train", "rmse", i*0.001, iteration=i)
    task.get_logger().report_scalar("test", "rmse", 0.001, iteration=i)

df = pd.DataFrame({'a': [0, 9, 8], 'b': [5, 6, 7]})
task.register_artifact('train data', df)

task.close()
