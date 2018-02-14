# Run the following score.py from the notebook to generate the web serivce schema JSON file
# Learn more about creating score file from here: https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy

def init():
    from sklearn.externals import joblib

    global model
    model = joblib.load('output/trainedModel.pkl')

def run(input_df):
    import json
    pred = model.predict(input_df)
    return json.dumps(str(pred[0]))

def main():
  from azureml.api.schema.dataTypes import DataTypes
  from azureml.api.schema.sampleDefinition import SampleDefinition
  from azureml.api.realtime.services import generate_schema
  import pandas

  df = pandas.DataFrame(data=[[380, 120, 76]], columns=['indicator1', 'NF1', 'cellprofiling'])

  # Check the output of the function
  init()
  input1 = pandas.DataFrame([[380, 120, 76]])
  print("Result: " + run(input1))
  
  inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}

  # Generate the service_schema.json
  generate_schema(run_func=run, inputs=inputs, filepath='output/service_schema.json')
  print("Schema generated")

if __name__ == "__main__":
    main()
