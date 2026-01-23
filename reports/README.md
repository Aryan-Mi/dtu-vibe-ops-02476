# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [ ] Create a git repository (M5)
* [ ] Make sure that all team members have write access to the GitHub repository (M5)
* [ ] Create a dedicated environment for you project to keep track of your packages (M2)
* [ ] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [ ] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [ ] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [ ] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [ ] Construct one or multiple docker files for your code (M10)
* [ ] Build the docker files locally and make sure they work as intended (M10)
* [ ] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [ ] Write unit tests related to the data part of your code (M16)
* [ ] Write unit tests related to model construction and or model training (M16)
* [ ] Calculate the code coverage (M16)
* [ ] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 36 ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

--- s252753, s254355, s204489, s226465, s175008 ---

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We opted for open-source frameworks for the front-end, deciding against using Streamlit. Our front-end stack utilizes React and Vite to deploy a Single Page Application (SPA). Specifically for 3D modeling within the front-end, we integrated framer-motion, three.js, and react-three-fiber into the project. ---

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

--- We used UV for managing our dependencies. The list of dependencies was auto-generated by running uv add as we introduce more dependencies necessary for the project. The command would automatically update the pyproject.toml file, which defines the dependencies our development environment needs. To get a complete copy of our development environment, one would have to do uv sync to install the dependencies listed in pyproject.toml.

Some of our training was done on the HPC, which uses the cuda version of pytorch instead of the CPU version. The dependencies for HPC are stored separately in hpc_pyproject.toml.txt, which one can copy over to pyproject.toml, then run uv sync to install the dependencies.
 ---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

--- From the mlops cookiecutter template, we have filled out the following core folders: .github (with workflows for CI/CD), configs (configuration files for models and training), data (raw and processed data), dockerfiles (containing api.dockerfile and train.dockerfile), docs (with mkdocs configuration), notebooks, reports (with figures subfolder), mlops_project (containing api.py, data.py, evaluate.py, model.py, train.py, visualize.py), and tests (with test_api.py, test_data.py, test_model.py). We also included the standard pyproject.toml, tasks.py, LICENSE, and .pre-commit-config.yaml from the template.

We removed the models/ folder from tracking and added it to .gitignore because we use DVC (models.dvc) to track trained models instead. We added several custom folders: .devcontainer for containerized development, .dvc for data version control configuration, hpc containing jobscripts for HPC cluster training, scripts with shell scripts for GCP setup and training job submission, vertex-ai for Google Cloud Vertex AI custom training job configuration, and .vscode to maintain consistent editor settings across the team. Additionally, we created extra Python modules like dataloader.py and subsample.py in the src folder to support our specific pipeline needs.

The frontend code is stored separately in another repository (https://github.com/Aryan-Mi/vibe-opsy) and does not follow the cookiecutter template structure. ---

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

--- We use Ruff for both linting and formatting of our Python code. The project's Ruff configuration enforces strict code quality standards to help the team consistently follow best coding practices. This includes enabling pycodestyle errors and warnings, requiring sorted imports, and enforcing naming conventions that follow PEP 8. For documentation, we have configured our project to use Google-style docstrings throughout the codebase. Ruff also incorporates checks from tools such as pyflakes, which helps catch errors and logical issues without executing the code. In addition, we have gradually introduced type hints using Python’s native typing syntax, improving readability and maintainability of the project.  ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

--- In total we have implemented 27 tests (28 test cases including parametrized). Primarily we are testing subsampling functionality and API endpoints as these are the most critical parts of our application, but also model architectures and data loading. Here’s the breakdown:
-Subsampling functionality (17 tests): data preprocessing, distribution maintenance, reproducibility, error handling.
-API endpoints (11 tests): health checks, prediction endpoints, image format handling, cancer detection logic.
-Model architectures (5 tests): input/output shape validation for BaselineCNN, ResNet, and EfficientNet.
-Data loading (2 tests): dataset initialization and comprehensive data loading validation. ---

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- The total code coverage of our code is approximately 35-40%, which includes all our source code. We are far from 100% coverage of our critical code and even if we were, then we still could not guarantee error-free code. Code coverage only measures which lines of code are executed during tests, not whether the logic is correct, whether edge cases are handled properly, or whether integration between components works correctly. A test can execute code with wrong inputs and still achieve 100% coverage. Additionally, coverage metrics do not account for real-world scenarios like data distribution shifts, model performance degradation, hardware-specific issues, or complex integration failures that only manifest when components interact in production environments. ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- In our project, we made use of both branches and pull requests. Each day that we worked on the project, we assigned specific tasks to each team member. For each assigned task, the person responsible would create a dedicated branch to implement and test their changes in isolation.Once the feature was implemented and locally tested, the person in charge of the feature would submit a pull request and ask one more group member to review. This ensured that at least two pairs of eyes had looked at the code before it was approved and squash-merged into the main branch, which also would help us to roll back in case we needed to. The team would then conduct a code review together, discussing the implementation and addressing any issues on the spot. Once the pull request was approved by the team, the branch would then be merged into main.  ---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- We made use of DVC to manage the dataset fed into our models, as well as the trained model artifacts. While the dataset itself is static, and thus we did not benefit too much from using DVC to manage our dataset, it is incredibly valuable for model versioning, as our model artifacts were constantly changing as more experiments were performed. DVC allowed us to conveniently version the raw data and resulting models, and distribute the latest model artifacts to the entire team. All data and model files were stored remotely in buckets on the Google Cloud Platform, with DVC configured to use these buckets as the storage backend, ensuring efficient access and collaboration without duplicating large files in version control. ---

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

--- We run continuous integration with GitHub Actions and keep it focused on quality checks, tests, and delivery. On every push and pull request to main, we run ruff for linting + formatting validation and pytest for unit tests with coverage reporting. Unit tests run with a matrix across Ubuntu, Windows, and macOS to catch OS-specific issues (currently on Python 3.12, but the setup can be extended to multiple Python versions if needed by simply providing a list of py versions).

In addition, we use CI/CD steps for containerized delivery with Docker images (training and api) that are built and validated on PRs, and are pushed to Google Cloud Artifact Registry only after merges to main. Deployment of the API to Google Cloud Run is gated so it only runs after the unit tests and the required image build succeed. This prevents broken builds from reaching production.

For speed, we rely on dependency caching via UV and GitHub Actions caching for Python tooling where relevant. ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- Experiments are configured using Hydra for flexible and reproducible individual runs. A base config.yaml file defines the general experiment settings, such as the batch size, image size and maximum number of epochs for training. When launching the train.py script, we can specify the model via a command-line argument (uv run src/mlops_project/train.py), which Hydra used to dynamically load the corresponding YAML files.
For hyperparameter tuning, we defined a sweeps.yaml file to set up sweeps in Wandb. When initiating a sweep, the training script would load both the standard Hydra configuration and the sweep parameters. The relevant sections of the Hydra config are then overwritten with the sampled values from the sweep, enabling efficient exploration of hyperparameter spaces while preserving the underlying structure and defaults from the base configuration.
 ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- The combination of hydra config files (config.yaml and model-specific YAMLs) and Weights and Biases (wandb) logging ensures that no information regarding the model parameters and hyperparameters used during each experiment were lost. The hydra configurations serve as the default settings for each experiment, with model-specific parameters loaded dynamically based on which model is chosen. While the wandb sweep configuration only defines the hyperparameter search space, information about each experiment, regardless of whether they are standalone experiments, or from a sweep, are logged comprehensively to wandb. In the overview page of the experiment the exact git commit hash can be found, capturing the precise code state, and the full command used to launch the experiment, enabling one to easily reproduce the experiments conducted. ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- In the first screenshot, we tracked the validation loss, validation accuracy, training loss, and training accuracy of various models in order to find which model was the best performing. These values allowed us to measure which models best predicted the presence of cancer in a given image of a skin lesion. We tested the models EfficientNet, a baseline CNN, and ResNet, and found that the optimal values were with EfficientNet. As can be observed in the second screenshot, once we established the model we wanted to use (EfficientNet), we finetuned the parameters in further experiments. Tracking the same variables, we found the parameters which optimized the model. 
`![various_model_tests](figures/various_model_tests.png)`
`![finetuning_model_tests](figures/finetuning_model_tests.png)`---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- For our project, we developed two specialized Docker images: one for training and one for API deployment.

Our training image (dockerfiles/train.dockerfile (https://github.com/Aryan-Mi/dtu-vibe-ops-02476/blob/main/dockerfiles/train.dockerfile)) packages the Python environment, source code, and Hydra configurations. It's designed for cloud execution on Vertex AI with commands like:
docker build -f dockerfiles/train.dockerfile -t train:latest .
docker run --name training-job -v ${PWD}/models:/models train:latest model=efficientnet training.max_epochs=20

Our API image (dockerfiles/api.dockerfile (https://github.com/Aryan-Mi/dtu-vibe-ops-02476/blob/main/dockerfiles/api.dockerfile)) contains DVC configurations for runtime model pulling and serves predictions via FastAPI. It's built and deployed automatically through our CI/CD pipeline to Google 

Cloud Run: docker build -f dockerfiles/api.dockerfile -t api:latest .
docker run -p 8080:8080 -e MODEL_NAME=EfficientNet api:latest
Our CI/CD pipeline builds images once and stores them in Google Artifact Registry. Deployments then pull these pre-built, tested images rather than rebuilding locally, ensuring consistency and faster deployments.
 ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- When we ran into bugs during experiments, we first checked the logs, which were available both locally and on Weights and Biases. Typically, the error messages themselves give clear hints as to what the issue is. To further narrow down the issues, we inserted inline breakpoints and executed our scripts in debug mode.

Additionally we experimented with the “torch-tb-profiler” package in-tandem with hydra to have an opt-in profiling config setup for the train method. However, enabling profiling significantly slowed down the training process, even when we heavily sub-sampled the dataset, while also hitting arbitrary limits and errors from the torch tb profiler package. Therefore, we ended up not merging the profiling setup in the final code.
 ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- We used the following GCP services in our project:
Compute Engine: Provides virtual machines (VMs) to run our training Docker containers for model development and experimentation.
Vertex AI: Managed machine learning platform used to train our deep learning models at scale with automated hyperparameter tuning and experiment tracking.
Cloud Build: Automatically builds and containerizes our Docker images for training, inference, and API deployment.
Cloud Run: Serverless platform hosting our FastAPI backend for real-time model inference and API requests.
Cloud Storage (Bucket): Data lake integrated with DVC for versioning and tracking training datasets, model checkpoints, and artifacts.
Networking: Manages communication and connectivity between all services (VMs, Cloud Run, Vertex AI, and storage).
VM Manager: Infrastructure management tool for provisioning and managing the Compute Engine VMs running our training workloads.
Cloud Logging: Centralized logging for monitoring and debugging all services across the pipeline.
Artifact Registry: Container registry for storing and managing Docker images built by Cloud Build. ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- We used Compute Engine primarily for model training by deploying a virtual machine with 2 CPU cores and 4GB RAM to execute our training Docker container. The VM instance was provisioned in the Belgium region to optimize for cost and latency.
Given our dataset size of approximately 3GB, the allocated compute resources proved insufficient for processing the full dataset in a reasonable timeframe. To address this bottleneck, we implemented a data slicing strategy that partitioned our dataset into smaller chunks, reducing individual training runs to approximately 30 minutes each. This approach allowed us to iterate faster during development and testing phases.
We later transitioned to Vertex AI for production training workflows, but Compute Engine remained valuable for experimentation and validation tasks where we needed direct control over the VM environment and custom container configurations.
 ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- `![gcp-question-19](figures/gcp-question-19.png)` ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- `![gcp-question-20](figures/gcp-question-20.png)` ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- `![gcp-question-21](figures/gcp-question-21.png)` ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- Yes we used Vertex AI to train our model in the cloud.
This was done by creating a train dockerfile and image, which runs on GCP. The training dockerfile pulls the training images using DVC from a GCP bucket.

However, we ended up moving away from Vertex and just use the HPCs at DTU instead. This is because training the full model on Vertex using 2 CPU cores and 4gb of ram took 3+ hours, while training on HPC’s gpuv100 queue, using 4 cores and 1 GPU with 32 GB of memory, could train the models in around 6 to 10 minutes, depending on the size of the EfficientNet.

 ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

--- Yes, we set up a simple back-end using FastAPI, in the api.py file. On startup, the app would initialize the image size expected by the model by reading the hydra configuration file and pull the trained ONNX model. The model will then be used for the inference endpoint.

The api has 2 simple endpoints:
(1) Health check: Returns the status of the server. If the server can be connected to, and the model is loaded, the server is considered to be healthy
(2) Inference: Using the ONNX model loaded in on startup, accepts an image as input, predicts a skin lesion type using the model and then returns a json dictionary with the predicted class, confidence, and whether the predicted class is considered a type of cancer to improve its interpretability to a layman user.
 ---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- Yes, we managed to deploy our API both locally and in the cloud.

For deployment, we wrapped our EfficientNet ONNX model into a FastAPI application (api.py). We first tested locally by building the Docker image using docker build -t skin-lesion-api -f dockerfiles/api.dockerfile . and running it with docker run -p 8080:8080 skin-lesion-api. This served the model successfully on localhost.

For cloud deployment, we use Google Cloud Run. We have a CI/CD pipeline (.github/workflows/api_deployment.yaml) that automatically deploys when api.py changes on the main branch. The pipeline waits for tests to pass, pulls model artifacts via DVC, builds the Docker image, pushes it to Google Container Registry, and deploys to Cloud Run.

To invoke the service, a user would call:


curl -X POST -F "file=@skin_image.jpg" https://skin-lesion-api-<region>.a.run.app/inference
This returns a JSON response with the predicted diagnosis, confidence score, and cancer classification. ---

### Question 25

> **Did you perform any functional testing and load testing of your API? If yes, explain how you did it and what**
> **results for the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For functional testing we used pytest with httpx to test our API endpoints and ensure they returned the correct*
> *responses. For load testing we used locust with 100 concurrent users. The results of the load testing showed that*
> *our API could handle approximately 500 requests per second before the service crashed.*
>
> Answer:

--- We performed unit testing but no load testing of our API. Our unit tests are discussed in Question 7 as well - our API test tests the FastAPI endpoints including health checks, prediction endpoints with valid/invalid inputs, image format handling (JPEG/PNG, RGB/grayscale), cancer detection logic, and response format validation.

To perform load testing, we would use Python to make many requests to the inference endpoint with sample images, gradually increasing the number of parallel requests while monitoring response times and identifying the breaking point where the API's performance degrades. We also considered using the hey library to perform load testing in a similar manner but from terminal.
 ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- We implemented an alert system for our skin-lesion-api Cloud Run service using Google Cloud's monitoring and logging infrastructure to proactively detect service failures and notify our team.
Cloud Logging collects all logs generated by our Cloud Run service, including requests, internal operations, and errors. When our service fails, it produces log entries with severity=ERROR containing timestamps, HTTP request details, status codes (like 500 Internal Server Error), and information about the specific Cloud Run service and revision.
We configured a log-based alert policy to continuously monitor these logs using the query: resource.type="cloud_run_revision" resource.labels.service_name="skin-lesion-api" severity=ERROR. This query filters for error logs originating specifically from our skin-lesion-api service.
When the number of matching log entries exceeds our predefined threshold within a specified time window (for example, more than zero error logs in a 5-minute period), the alert policy triggers. Notifications are then sent through our configured email, aka send Marvin an email.
---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- Most of our service costs were related to working with Google Cloud and running our model there, but overall the costs were not very high. For our Google Cloud usage, the Compute Engine, Networking, VM Manager, Vertex AI, Cloud Storage, Artifact Registry, Cloud Run cost under $10. For Github Actions, we used over 1000 credits, which was well within our 3000 credit limit. Working in the cloud was new for some members, while others had familiarity. It was a source of frustration at times, especially when it came to setting up the containers to run Google Cloud. There was also a member who felt that Google Cloud is an overengineered sack of sh*t, and preferred other cloud providers. ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- We implemented a frontend for our API. We did this because we wanted the user to be able to easily submit images for consideration of our model and receive feedback in a straightforward manner.

The frontend was decided via a frontend competition where each member presented their idea and the best was selected via group vote - Aryan won.

The frontend that was selected was implemented using react, because the idea was to render an actual 3D model (.glb) file of an old macintosh computer. And react made sense for this scenario, because the huge ecosystem made it easy to set up all of the interactive parts relatively easily in only one day. The front-end was then deployed on cloudflare, since the developer experience is much better for deploying a SPA react app.
Link to the website: https://vibe-opsy.aryan-mi.workers.dev/ ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- Our system begins with local development where code is managed in a GitHub repository, integrated with DVC for data version control and Wandb for experiment tracking. When code is pushed to GitHub, it automatically triggers our CI/CD pipeline through GitHub Actions. This pipeline runs unit tests with pytest, performs code quality checks using ruff, and builds Docker images for any PR on main and pushes them to GCP Registry for commits in main. All tests must pass before proceeding to deployment.
The data pipeline starts with raw images stored in GCS buckets, version-controlled through DVC. Our data processing includes validation, transformation, and subsampling steps to prepare training data. Model training is configured using Hydra for managing hyperparameters, which also supports Weights and Biases sweeps, with all experiments logged to Wandb for tracking metrics, losses, and model performance comparisons.
Trained models are stored in DVC-managed buckets and converted to ONNX format for optimized inference. The deployment architecture uses Docker containers that package the FastAPI application with the trained model, that can be triggered by change detecting a in api.py file or manually in GitHub actions.
The inference service exposes a REST API with endpoints for health checks and predictions. When users submit skin lesion images, the API preprocesses them (resizing to 224x224, normalization, RGB conversion), runs inference using the ONNX model, and returns predictions including the disease class, confidence scores, probabilities for all seven classes, and a binary cancer/non-cancer classification.
 `![system-architecture](figures/system-architecture.png)` ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- We faced many challenges throughout the process of building our project. The first major challenge we had was getting DVC to work properly for each member as it was a concept that was relatively new to each of us. The issue was that we were not correctly pulling the actual data files from remote storage. We were able to resolve this by reconfiguring our dvc setup for the GCP bucket rather than default storage. We encountered a deployment challenge with Vertex AI, specifically related to accessing the necessary storage bucket. This required us to configure a resource service account and grant it the appropriate permissions for bucket access. Initially, we considered using Google's serverless functions for the backend, but further investigation showed that Cloud Run is the preferred direction, primarily due to its containerized architecture and it also allows us to load the ONNX model on build instead of during each request from a bucket.

We faced several Wandb challenges, particularly when it came to the model registry process. We initially planned to store our trained models in a DVC bucket and then register them with Wandb for model versioning and management. However, we faced technical difficulties integrating these two systems, specifically around artifact linking, collection creation, and automated workflows between DVC and Wandb's model registry. Due to time constraints and the complexity of the integration, we pivoted to using Wandb primarily for experiment tracking and metrics logging, while keeping our models in DVC. This meant we lost the benefit of Wandb's built-in model versioning features and had to rely on DVC for version control instead.
 ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
