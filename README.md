# AML-DevOps-Workshop
> ML/DevOps Hands on workshop

## Agenda

- 09:30-10:00 Workshop overview, scope, expectations
- 10:00-10:50 Dev environment setup: Azure ML service Workspace and Azure Notebooks. Authenticate, prepare compute (Azure ML Compute)
- 11:00-11:50 Train first DL model on Azure Notebooks using Azure ML Compute
- 13:00-14:20 Distributed training with Horovod on AML Compute, explore AML Workspace
- 14:30-15:20 Create container images, deploy to Azure Container Instance (and/or Azure Kubernetes Service)
- 15:30-16:50 Dev environment setup: Use GitHub Desktop, Azure DevOps(create DevOps account, Organization), create from Azure ML template, customize Build Pipeline, customize Release Pipeline
- 17:00-17:50 Questions and answers

## ML Track

- **09:30-10:00 Workshop overview, scope, expectations**
  - Process flow and architecture ([pdf](https://github.com/dem108/AMLWorkshop-IotEdge-DevOps/blob/master/doc/decks/Microsoft%20AI%20Architecture%20one-slider-EN-v20190513.pdf))
  - DevOps pipeline ([pdf](https://github.com/dem108/AMLWorkshop-IotEdge-DevOps/blob/master/doc/decks/DevOps-ML-IotEdge-pipeline-flow-v20190513.pdf))

- **10:00-10:50 Dev environment setup: Azure ML service Workspace and Azure Notebooks, authenticate, prepare compute (Azure ML Compute)**

    1. Install
        - [Visual Studio Code](https://code.visualstudio.com/)
        - [GitHub Desktop](https://desktop.github.com/)
        - (optional) Internet browser of your choice (Edge is fine, Chrome is also good)
    1. Check Azure subscription
        - All attendee should be able to sign in
    1. [Create an AML service workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace)
        - region: East US
        - resource group: new (one per person for practice)
        - after creation, check `Usage + quotas`, Standard NC Family vCPUs: should have 100+ available dedicated cores for this workshop (e.g., 5 people * 6 cores * 4 nodes = 120 cores)
    1. (optional) Add users in `Access Control (IAM)`
        - FYI: [Manage users and roles - create custom roles](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-assign-roles#create-custom-role)
    1. From Azure ML service Workspace `Overview` tab, click `Download config.json`, save locally.
    1. Set up Notebook environment. In this workshop, use Option 1 to practice.
        - Option 1: using Azure Notebooks
            - [Import the AML sample from GitHub](https://docs.microsoft.com/en-us/azure/notebooks/create-clone-jupyter-notebooks#import-a-project-from-github)
                - GitHub repo to import: https://github.com/royboy0416/AML-DevOps-Workshop
                - private (if needed)
        - Option 2: using Notebook VMs from Azure ML service Workspace
            - go to `Notebook VMs`, create a new VM (STANDARD_D3_V2)
            - Click `JupyterLab`, click `Terminal`
            - From the current directory `/mnt/azmnt/code/Users/`, cd <USERNAME> (or mkdir if needed), git clone with `git clone https://github.com/royboy0416/AML-DevOps-Workshop
            - Note that the config.json is already automatically added to `/mnt/azmnt/`, you do ***not*** need to upload it manually.
            - From `Notebook VMs`, click `Jupyter`, and you can run notebooks there
            ![](https://raw.githubusercontent.com/dem108/AMLWorkshop-IotEdge-DevOps/master/doc/images/setup-notebook-vm-jupyter-notebook.jpg)

    1. Create Azure ML Compute: To do that, open `0.configurations.ipynb`

        - Run the first cell. By running this cell, a new folder will be created in your project.
            ```python
            import os
            os.makedirs('aml_config')
            ```
        - Upload the config.json file that you have downloaded in the above. 
        - Then, add a cell and run following script to load the config.json and authenticate.
            ```python
            from azureml.core import Workspace
            ws = Workspace.from_config()
            ```
            ![](https://raw.githubusercontent.com/dem108/AMLWorkshop-IotEdge-DevOps/master/doc/images/authenticate-workspace.jpg))
        - Proceed to create Azure ML Compute
            - `cpucluster` STANDARD_D2_V3, 0 to 4 nodes
            - `gpucluster` STANDARD_NC6, 0 to 4 nodes

- **11:00-11:50 Train first DL model on Azure Notebooks using Azure ML Compute**

    1. Open sample notebook `1.train-hyperparameter-tune-deploy-with-keras-part1.ipynb` under `code` folder (find [this notebook](https://github.com/royboy0416/AML-DevOps-Workshop/blob/master/code/1.train-hyperparameter-tune-deploy-with-keras-part1.ipynb) from your notebook environment)
    1. Run (before `run.wait_for_completion()` cell)
    1. Monitor the Jupyter widget, and the Workspace (from Azure Portal - check Experiment and Compute)
    1. Additionally, note that files in `./outputs` and `./logs` are automatically uploaded to the Workspace. Tensorboard logs should also be saved in this `./logs`. Refer to [how to train models](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-ml-models#single-node-training) and [TensorBoard integration sample](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/training-with-deep-learning/tensorboard/tensorboard.ipynb).
    1. Try to understand how the model files are moving, from AML Compute, to Workspace, to local environment.
    1. Continue running the notebook and try hyperparameter tuning.
        - Set `max_concurrent_job` parameter to the maximum number of nodes in your Azure ML Compute cluster.
        - Run, monitor the Jupyter widget and Azure Portal (AML service Workspace), evaluate the results
            > Note: Generally when you open the Notebook, you can see the last run results of the code cells, but Jupyter widget results are not shown. So in order to review last Widget run status without running the experiment again, you should find and load the run before using the widget. Sample notebook to do this is [here](https://github.com/dem108/AMLWorkshop-IotEdge-DevOps/blob/master/notebooks/Check-Jupyter-widget-for-a-specific-run.ipynb). 

- **13:00-14:20 Distributed training with Horovod on AML Compute, explore AML Workspace**

    1. Open sample notebook `2.distributed-pytorch-with-horovod.ipynb` under `code` folder (find [this notebook](https://github.com/royboy0416/AML-DevOps-Workshop/blob/master/code/2.distributed-pytorch-with-horovod.ipynb) from your notebook environment)
    1. Run all: consider using 4 nodes when available instead of 2 as `node_count`.
    1. Questions and answers, or proceed to the next step.

- **14:30-15:20 Create container images, deploy to Azure Container Instance (and/or Azure Kubernetes Service)**

    1. We will continue from morning's sample. Open sample notebook `3.train-hyperparameter-tune-deploy-with-keras-part2.ipynb`, and run creating container image and deploying to ACI.
    1. Explore Workspace from Azure Portal.
    1. Refresh the concepts of MLOps from [concept-model-management-and-deployment](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-model-management-and-deployment)

    * If time permits, try below contents in addition:

        - [Build 2019 updates](https://azure.microsoft.com/en-us/blog/new-azure-machine-learning-updates-simplify-and-accelerate-the-ml-lifecycle/): New Azure Machine Learning updates simplify and accelerate the ML lifecycle
        - [visual-interface (preview)](https://docs.microsoft.com/en-us/azure/machine-learning/service/ui-tutorial-automobile-price-train-score)
        - [automated ml with GUI (preview)](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-create-portal-experiments)
        - [interpretability-explainability](https://docs.microsoft.com/en-us/azure/machine-learning/service/machine-learning-interpretability-explainability)
        - [onnx](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-onnx)
        - [fpga](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-accelerate-with-fpgas)
        - [pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-ml-pipelines)
        - [security](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-enterprise-security)
        - [custom vision](https://customvision.ai)

    * Running AML SDK on Azure Databricks
        - Set up Azure Databricks using this [guide](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#azure-databricks)
        - Create a cluster, and import the [sample notebook](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/azure-databricks/Databricks_AMLSDK_1-4_6.dbc).
        - Install `azureml-sdk[automl_databricks]` if needed.
        - Run samples.
            - For Automated ML sample, set `max_concurrent_iterations` to the number of worker nodes.

    * Check out [MLOps](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-model-management-and-deployment). 

## DevOps Track

- **15:30-16:50 Dev environment setup: Use GitHub Desktop, Azure DevOps(create DevOps account, Organization), create from Azure ML template, customize Build Pipeline, customize Release Pipeline**

    * Check out [MLOps](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-model-management-and-deployment). 

    1. Create an Azure DevOps account from [DevOps start page](https://azure.microsoft.com/en-us/services/devops/?nav=min) - `Start Free`
        What you also create is an `organization`. Note the organization name created.

    1. We will use this quick starter - [Demo Generator](https://azuredevopsdemogenerator.azurewebsites.net/?name=azure%20machine%20learning)
    
        1. Alternative way to do this is [LearnAI_Azure_ML](https://github.com/Azure/LearnAI_Azure_ML/tree/master/devops), which helps you with step-by-step approach to create the pipeline from scratch. We will ***not*** use this in this workshop.

    1. From the Demo Generator, choose the template, `Azure Machine Learning`. It's under `DevOps Labs` tab. Choose your `organization`, and specify the project name to create. 
        ![](https://raw.githubusercontent.com/dem108/AMLWorkshop-IotEdge-DevOps/master/doc/images/devops-generator-01-create-with-azureml-template.jpg)
        ![](https://raw.githubusercontent.com/dem108/AMLWorkshop-IotEdge-DevOps/master/doc/images/devops-generator-02-create-with-azureml-template.jpg)

    1. Explore Repos
        1. Edit config.json under aml_config. You can obtain the content for this file from AML service Workspace `Overview` from Azure Portal.
        1. Notice that editing this file lets to commit it to master branch, which will initiate the Build Pipeline. It will fail and we'll fix the issue in the following steps.
            - You can alternatively commit it to another branch, and merge it later into the master branch. A general git practice.

        > Note: Instead of keeping sensitive files in Repo you could use `Secure File` feature from Azure Pipelines. A sample guidance is [here](https://github.com/Azure/LearnAI_Azure_ML/blob/master/devops/01-Build.ipynb). More details on Secure Files [here](https://docs.microsoft.com/en-us/azure/devops/pipelines/library/secure-files?view=azure-devops).
    1. Explore Pipelines
    1. Edit Build Pipeline `DevOps-for-AI-CI`
        1. Starting from `Create or Get Workspace`, specify the Azure subscription to use, and authorize.
        1. Save and Queue.
        1. Monitor the run, and fix any outstanding issues.

        > Note: We are using Azure CLI Authentication now. Check out other ways to [authenticate](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azure-ml.ipynb).

    1. Open `Deploy Webservice` Release Pipeline. Notice that releases were automatically initiated but failed.
    
    1. Click `Edit` for the Release Pipeline. Check out `Pre-deployment conditions`, and `Post-deployment conditions` for each stage.

        1. In the `Prod - Deploy on AKS` stage, check out `Gates`. See what deployment gates can be added.

        1. Click `1 job, 4 tasks` under `QA - Deploy on ACI` stage.

        1. Specify Azure subscriptions to use for deployment and test.

        1. Continue to the `Prod - Deploy on AKS` stage and do the same.
