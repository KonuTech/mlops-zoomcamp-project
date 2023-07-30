git push -u origin main
git submodule add https://github.com/DataTalksClub/mlops-zoomcamp.git mlops-zoomcamp
cd /home/konradballegro/mlops-zoomcamp/03-orchestration
ls
cp 3.5 ..
cp 3.5 -r ..
cd
ls
cd orchestration/
ls -lah
python3 orchestrate.py
cd
ls -lah
rm -r mlops-zoomcamp/
git status
git init
git status
git clone https://github.com/DataTalksClub/mlops-zoomcamp.git
git init
git status
git add orchestration/
git status
git add .prefect/
git commit "third commit"
git commit -m "third commit"
git push -u origin main
cd orchestration/
touch create_gs_bucket_block.py
vim create_gs_bucket_block.py 
cat create_
cat create_gs_bucket_block.py 
prefect block ls
gcloud auth login
gcloud auth application-default login
gcloud config set project ny-rides-konrad
gcloud
gcloud config
gcloud config list
ls
touch upload_folder_to_gs.py
vim upload_folder_to_gs.py 
cat upload_folder_to_gs.py 
ls
python3 upload_folder_to_gs.py 
python3 upload_folder_to_gs.py mlops-zoomcamp data data
touch orchestrate_gs.py
vim orchestrate_gs.py 
cat orchestrate_gs.py
prefect deployment build -n mlops-gs-test -p default-agent-pool -q mlops-gs orchestrate_gs.py:main_flow
pip list
pipenv install prefect_gcp
cd
ls -lah
pipenv install prefect_gcp
pip list
cd orchestration/
ls -lah
prefect deployment build -n mlops-gs-test -p default-agent-pool -q mlops-gs orchestrate_gs.py:main_flow
prefect deployment build -n mlops-gs-test -p default-agent-pool -q mlops-gs orchestrate_gs.py:main_flow_gs
prefect worker start -p zoompool
python3 orchestrate_gs.py
prefect workers
prefect --help
prefect work-pool
prefect work-pool create zoompool
pip list
python3 orchestrate.py
python3 orchestrate_gs.py
touch test_load_from_gs.py
vim test_load_from_gs.py 
ls
python3 test_load_from_gs.py 
pipenv uninstall prefect_gcp
pip uninstall prefect_gcp
python3 orchestrate_gs.py
python3 orchestrate.py
prefect deployment build -n mlops-gs-test -p default-agent-pool -q mlops-gs orchestrate_gs.py:main_flow_gs
python3 orchestrate_gs.py
git status
git add .gitignore .prefect
cd
git add .gitignore .prefect
git add Pipfile
git add create_s3_bucket_block.py
git add mlglow.db
git add mlflow.db
git add main_flow_gs-deployment.yaml
git add orchestration/
git status
git add .gitconfig
git commit -m "third commit"
git push -u origin main
pip list
source .virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
git add requirements.txt web-service/
git status
git add .prefect/
git status
git commit -m "sixth commit"
git push -u origin master
git status
git push
pipenv lock
mkdir scoring-batch
ls
rm scoring-batch/
mkdir batch-scoring
ls
rm mlruns
rm -r mlruns
git status
git add web-service/
git add scoring-batch/
git status
git commit -m "seventh commit"
git push -u origin main
git status
pipenv
pipenv --dev
pipenv lock --dev
pipenv --graph
pipenv graph
pipenv check
pipenv
pipenv lock
pipenv --virtual
pipenv --venv
pipenv --envs
pipenv update
pipenv install notebook
pip freeze > requirements.txt 
cd orchestration/
ls
python3 orchestrate_gs.py 
cd
jupyter notebook
pipenv shell
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd notebooks/
ls
cd outputs/
ls -lah
ls -las
source .virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
python3 orchestration/orchestrate_gs.py 
cd orchestration/
python3 orchestrate_gs.py 
ls
python3 lr.py
python3 orchestrate_gs.py 
python3 lr.py
cd
git status
git add .bash_history .prefect/ lr.py
git status
git add orchestration/
git status
ls
rm mlflow.db 
ls
git status
git add temp
git status
git commit -m "fourth commit"
git branch -M main
git push -u origin main
cd orchestration/
ls
rm -rf mlruns
rm mlflow.db 
git status
git add orchestrate
cd
git status
git add orchestration/
git status
git commit -m "fifth commit"
git push -u origin main
cd orchestration/
python3 orchestrate_gs.py 
cd
git clone https://github.com/DataTalksClub/mlops-zoomcamp.git
pip list
pip freeze > requirements.txt
cd mlops-zoomcamp/04-deployment/web-service-mlflow/
ls
flask --run predict.py 
flask --app predict.py 
flask run
python3 predict.py 
cd
cd web-service/
ls -lah
python3 test.py
python3 predict.py 
python3 test.py
cd
git status
git add requirements.txt 
git add requirements_old.txt 
git add Pipfile 
git add mlops-zoomcamp/
git add scoring-batch/
git add web-service/
git add .prefect/
git status
git commit -m "eighth commit"
git push -u origin main
git status
source .virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
cd mlops-zoomcamp/04-deployment/web-service-mlflow/
ls
python3 test.pt
python3 test.py
pip list
python3 test.py
pip list
python3 test.py
cd 
mkdir web-service
cd web-service
ls
ls -lah
python predict.py
source .virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
prefect server start
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
cd
ls
mkdir notebooks
ls
ls-lah
ls -lah
cd notebooks/
mkdir outputs
mkdir inputs
cd
cd orchestration/
python orchestrate_gs.py 
cd
git status
git add notebooks
git add .prefect/
git add mlops-zoomcamp/
git add orchestration/
git commit -m "nineth commit"
git status
git push
git status
htop
git status
git add .bash_history 
git add mlops-zoomcamp/
git add notebooks/
git status
git add notebooks/
git status
git add .gitignore 
git commit -m "tenth commit"
git push
git checkout -- /home/konradballegro/notebooks/outputs/predictions.parquet
git status
git push
git checkout /home/konradballegro/notebooks/outputs/predictions.parquet
git status
git add notebooks/
git commit -m "tenth commit"
git push
cd notebooks/
ls
cd outpus
cd outputs
ls -lah
ls
git status
git rm notebooks/outputs/predictions.parquet
git push
git reset HEAD~
git status
git add predictions.parquet 
cd
git push
git status
git restore notebooks/outputs/predictions.parquet
git push
git restore --staged notebooks/outputs/predictions.parquet
git status
git restore notebooks/outputs/predictions.parquet
git status
git rm notebooks/outputs/predictions.parquet
git status
git push
git reset HEAD~
git status
git add notebooks/starter.ipynb 
git commit -m "tenth commit"
git status
git push
pipenv shell
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pipenv install --dev pylint
ls
pipenv lock
pipenv lock --three
pylint --version
pip list
pip freeze > requirements.txt
pylint --recursive=y
which pylint
python
which python
pip list env
pip env list
conda env list
pip env list
ls ..
ls -lah ..
ls ubuntu
ls ../ubuntu
ls -lah ../ubunut
ls -lah ../ubuntu
pipenv shell
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pylint --recursive=y
pylint --recursive=y /home/konradballegro/mlops-zoomcamp/06-best-practices/code/
pip list
pipenv install --dev black isort
black --diff /home/konradballegro/mlops-zoomcamp/06-best-practices/code/tests/model_test.py
black --diff /home/konradballegro/mlops-zoomcamp/06-best-practices/code/model.py
black --diff /home/konradballegro/orchestration/orchestrate_gs.py
black /home/konradballegro/orchestration/orchestrate_gs.py
pylint /home/konradballegro/orchestration/models
pylint /home/konradballegro/orchestration
make
touch Makefile
make
make run
git status
git add .
git status
git reset
git status
git add Makefile 
git add .bash_history 
git add .gitignore 
git add Pipfile 
git add mlops-zoomcamp/
git add orchestration/
git add requirements.txt 
git status
git add .gitignore 
git commit -m "eleventh commit"
git push
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
jupyter notebook
touch env
make install-pyspark
nano .bashrc
spark-shell
source .bashrc
which java
make add-java 
which java
java --version
export JAVA_HOME="${HOME}/spark/jdk-11.0.2"
export PATH="${JAVA_HOME}/bin:${PATH}"
java --version
which java
make add-java
source .bashrc
which java
make add-java
which java
make add-java
which java
source ~/.bashrc
which java
gitenv shell
pipenv shell
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
ls -lah
ls
ls -a
ls -la
ls
make install-java
git status
which JAVA
java --version
git status
make install-spark
git status
spark shell
spark-shell
pip list
spark
source .bashrc
spark-shell
source .bashrc
spark-shell
./bin/spark-shell
/usr/local/spark/bin/spark-shell
printenv
printenv | grep PATH
Spark
spark
spark-shell
cd %SPARK_HOME%/bin
echo $PATH
echo $PYTHONPATH
echo $SPARK_HOME 
java --version
export JAVA_HOME="${HOME}/spark/jdk-11.0.2"
export PATH="${JAVA_HOME}/bin:${PATH}"
java --version
source .bashrc
printenv
which java
java --version
export JAVA_HOME="${HOME}/spark/jdk-11.0.2"
export PATH="${JAVA_HOME}/bin:${PATH}"
java --version
which java
source .bashrc
which java
make install-java
which java
export JAVA_HOME="${HOME}/spark/jdk-11.0.2"
export PATH="${JAVA_HOME}/bin:${PATH}"
which java
java --version
export SPARK_HOME="${HOME}/spark/spark-3.3.2-bin-hadoop3"
export PATH="${SPARK_HOME}/bin:${PATH}"
spark-shell
which spark
spark --version
spark-version
spark-shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
spark-shell
spark-version
which spark
which java
make add-java 
which java
java --version
make add-java 
java --version
make add-java 
make add-java
which java
export JAVA_HOME="${HOME}/spark/jdk-11.0.2"
which java
export PATH="${JAVA_HOME}/bin:${PATH}"
which java
java --version
echo java --version
which java
make add-java
source .bashrc
make add-java
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
make add-java
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
make add-java
which java
java --version
make pyspark-install 
spark-shell
export JAVA_HOME="${HOME}/spark/jdk-11.0.2" && export PATH="${JAVA_HOME}/bin:${PATH}" && java --version && which java
spark-sehll
spark-shell
make spark-add 
spark-shell
export SPARK_HOME="${HOME}/spark/spark-3.3.2-bin-hadoop3"
export PATH="${SPARK_HOME}/bin:${PATH}"
spark-shell
make spark-add 
spark-shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
spark-shell
make spark-add 
spark-shell
make java-add 
spark-shell
make spark-add 
spark-shell
printenv
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
spark-shell
make
make java-add
spark-shell
make java-install 
spark-shell
make spark-install
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
spark-shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
touch .env
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
spark-shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
jupyter notebook
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
printenv
shell-script
spark-shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
printenv
jupyter notebook
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
git add .bash_history 
git add .bashrc 
git add .gitignore 
git add Makefile 
git add mlops-zoomcamp/
git status
git add .gitignore 
git status
git add .gitignore 
git add notebooks/spark_test.ipynb 
git add .gitignore 
git status
git commit -m "installed spark"
git push
git status
git add .env
git add .gitignore
git commit -m "added .env"
git push
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
spark-shell
ls
jupyter notebook
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
make pyspark-install 
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
ls
make
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
jupyter notebook
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
make
make spark-install 
make pyspark-install 
make
java which
which java
spark-shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
spark-sehll
spark-shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
spark-shell
which java
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
jupyter notebook
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
jupyter notebook
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
make setup
spark-shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
/home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/python3.8
jupyter notebook
git status
git ls -lah notebooks
ls -lah notebooks
git status
git add .bash_history 
git add .bashrc
git add .env
git add Makefile 
git add mlops-zoomcamp/
git add notebooks/
git status
git commit -m "installed spark"
git push
ls
mkdir flows
mkdir scripts
ls
ls scripts
mkdir scraper
cd scripts
cd scraper
ls
ls modules
ls
python3 main.py
pip show lxml
pipenv shell
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pipenv install lxml
pwd
cd /home/konradballegro/scripts/scraper
ls
python3 main.py
cd ..
cd
git status
git add Pipfile 
git add mlops-zoomcamp/
git add scripts/
git commit -m "added scraper"
git push
python3 main.py
cd /home/konradballegro/scripts/scraper
ls
python3 main.py
cd inputs
ls
cd header.txt > header_copy.txt
python3 main.py
cd ..
python3 main.py
cd
git status
git add .bash_history 
git ad mlops-zoomcamp/
git add mlops-zoomcamp/
git add scripts
git status
git add mlops-zoomcamp/
git status
git commit -m "added scraper"
git push
python3 /home/konradballegro/scripts/scraper/main.py
cd /home/konradballegro/scripts/scraper/
python3 main.py
cd
git status
git add .bash_history 
git add mlops-zoomcamp/
git add scripts/
git status
git commit -m "scraper - added English header"
git push
ls
cd /home/konradballegro/scripts/scraper/modules/scrapers
black .
cd ..
ls
cd scrapers/
ls
cd ..
ls
cd ..
ls
python3 main.py
cd /home/konradballegro/scripts/scraper/modules/scrapers
ls
pylint .
isort .
pylint .
cd ..
ls
python3 main.py
cd
git status
git add scripts/
git commit -m "scraper - update"
git push
cd /home/konradballegro/scripts/scraper/modules/scrapers
ls
cd ..
ls
cd ..
python3 main.py
cd
git status
git add scripts
git commit -m "scraper - update"
git push
git status
git add scripts/
git commit -m "scraper - update"
git push
git status
cd /home/konradballegro/scripts/scraper
ls
python3 main.py
git status
git add scripts
cd
git status
git add scripts
git commit -m "scraper - added appending of new rows without duplicates"
git pus
git push
cd /home/konradballegro/scripts/scraper
ls
python3 main.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
htop
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
htop
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd /home/konradballegro/scripts/scraper/outputs/data
ls
ls -lah
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
git add Pipfile 
git add notebooks/
git commit -m "added notebooks"
git push
git status
git add requirements.txt 
git add scripts/scraper/inputs/manufacturers.txt
git status
git .bash_history
git add .bash_history
git commit -m "added notebooks"
git push
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd /home/konradballegro/scripts/scraper
ls
python3 main.py
ls
cd outputs
ls -lah
cd data
ls -lah
cd
spark-shell
jupyter notebook
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
pip env install --dev seaborn
pipenv install --dev seaborn
pip freeze > requirements.txt
pipenv install --dev ydata-profiling
git status
cd /home/konradballegro/notebooks
ls
mkdir ouputs/reports
ls
mkdir outputs
cd outputs/
ls
mkdir reports
htop
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd /home/konradballegro/notebooks/outputs
ls
mkdir data
pip list
cd
git status
git add .bash_history notebooks/regression.ipynb notebooks/variable_selection.ipynb
git status
git commit -m "added variable selection"
git push
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
htop
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
jupyter notebook
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
htop
pienv shell
pipenv shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
git status
git add .bash_history notebooks/variable_selection.ipynb notebooks/champion_selection.ipynb notebooks/explainer.ipynb notebooks/outputs/data/
git status
git commit -m "trained xgboost regressor"
git push
pipenv --dev install explainerdashboard
pipenv install --dev explainerdashboard
pip list
pip freeze > requirements.txt
git status
git add Pipfile notebooks/champion_selection.ipynb notebooks/explainer.ipynb notebooks/variable_selection.ipynb requirements.txt
git commit -m "trained diffrent models"
git push
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
jupyter notebook
ls
pipenv shell
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
htop
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd /home/konradballegro/notebooks/outputs/reports
ls -lah
touch test
ls -lah
rm test
ls
cd /home/konradballegro/notebooks/outputs/models
ls -lah
cd /home/konradballegro/notebooks/outputs/reports
ls -lah
cd /home/konradballegro/notebooks/outputs/data
ls -lah
cd /home/konradballegro/notebooks/outputs/models
ls -lah
cd /home/konradballegro/notebooks/outputs/reports
ls -lah
/home/konradballegro/notebooks/outputs/models
ls -lah
/home/konradballegro/notebooks/outputs/models
cd //
cd /home/konradballegro/notebooks/outputs/models
ls
ls -lah
/home/konradballegro/notebooks/outputs/models
cd /home/konradballegro/notebooks/outputs/models
ls -lah
cd
git status
git add .bash_history mlops-zoomcamp notebooks/champion_selection.ipynb notebooks/explainer.ipynb notebooks/variable_selection.ipynb 
git status
git commit -m "trained a regressor v01"
git push
cd /home/konradballegro/notebooks/outputs/models
ls -lah
cd /home/konradballegro/notebooks/outputs/reports
ls -lah
git add notebooks/explainer_xgb.ipynb notebooks/outputs/reports/xgb_explainer.html
git status
cd 
git status
git add notebooks/outputs/reports/xgb_explainer.html notebooks/explainer_xgb.ipynb notebooks/outputs/models/ notebooks/outputs/reports/xgb_explainer.html
git status
git commit -m "trained a regressor v01"
git push
git status
git add notebooks/outputs/data/offers.csv
git commit -m "added raw input"
git push
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
jupyter notebook
pipenv shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd orchestration/
python3 orchestrate_gs.py
ls -lah
python3 orchestrate_training.pt
python3 orchestrate_training.py
git status
cd
git status
git add .bash_history .prefect/prefect.db notebooks/champion_selection.ipynb notebooks/explainer_xgb.ipynb orchestration/*
git status
git reset HEAD~
git status
git add orchestration/data/*
git status
git add orchestration/models/
git add orchestration/modules/
git add orchestration/utils/
git status
git add orchestration/orchestrate_training.py
git add .bash_history notebooks/champion_selection.ipynb notebooks/explainer_xgb.ipynb 
git commit -m "added orchestrate_training"
git push
git pull
git pull force
git pull
git commit -m "added orchestrate_training"
git push
git stash
git pull
git push
git stash apply
git status
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd /home/konradballegro/notebooks/outputs/data
ls -lah
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
jupyter notebook
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
prefect start
prefect server
prefect server start
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pipenv shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
ls
cd orchestration/
ls -lah
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd orchestration/
ls -lah
cd
git status
git add orchestration/otomoto_training.py
git add orchestration/otomoto_scraping.py orchestration/models/xgb copy.model orchestration/otomoto_scraping_flow-deployment.yaml orchestration/.prefectignore orchestration/data/inputs/offers.csv
git add orchestration/otomoto_scraping.py orchestration/otomoto_scraping_flow-deployment.yaml orchestration/.prefectignore orchestration/data/inputs/offers.csv
git status
git commit -m "otomoto_training changes"
git push
git reset HEAD~
git status
git add orchestration/otomoto_scraping.py orchestration/otomoto_scraping_flow-deployment.yaml
git add orchestration/data/outputs/offers_filtered.csv
git commit -m "otomoto_training changes"
git push
git pull
git merge
git status
git reset HEAD~
git status
git pull
git pull -force
git pull --force
git stash
git pull
git status
git add orchestration/.prefectignore
git status
git stash
git pull
git add orchestration/data/inputs/offers.csv orchestration/otomoto_scraping.py orchestration/otomoto_scraping_flow-deployment.yaml
git stash
git pull
git status
git add orchestration/otomoto_training.py
git add orchestration/config/
git status
orchestration/otomoto_training_copy.py
git add orchestration/data/inputs/manufacturers_copy.txt
git add orchestration/otomoto_training_copy.py
git status
git add orchestration/models/xgb_copy.model
git status
git commit -m "otomotmot_training changes"
git push
git status
git add orchestration/otomoto_training.py
git commit -m "otomotmot_training changes"
git push
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
git add .bash_history notebooks/explainer_xgb.ipynb orchestration/logs/app.log orchestration/otomoto_scrap_data.py orchestration/otomoto_scrap_data_flow-deployment.yaml orchestration/otomoto_training.py
git status
git add orchestration/.prefectignore orchestration/data/inputs/manufacturers copy.txt orchestration/otomoto_scraping.py orchestration/otomoto_scraping_flow-deployment.yaml
git status
git commit -m "added first vesrion of training flow"
git push
git status
cd /home/konradballegro/orchestration/data/outputs
ls -lah
cd /home/konradballegro/orchestration/models
ls -lah
htop
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd /home/konradballegro/orchestration/outputs/data
ls -lah
cd
git statys
git status
git add scripts
git status
git add orchestration/
git status
git add .bash_history 
git add .prefect/prefect.db 
git status
git commit -m "added first tasks to otomoto training flow"
git push
git status
git add orchestration/
git status
git add requirements_old.txt 
git commit -m "added otomoto scraping flow"
git push
git status
git add orchestration/
git add temp
git status
git commit -m "changes
"
git push
git status
git add orchestration/data/inputs/manufacturers.txt
git add orchestration/logs/app.log
git add orchestration/data/inputs/manufacturers copy.txt
git status
git add orchestration/data/inputs/manufacturers copy.txt
git status
git add .gitignore 
git status
git commit -m "changes"
git push
cd orchestration/
ls
black otomoto_training.py
prefect deployment build -n otomoto-scraping -p default-agent-pool -q otomoto-scraping otomoto_scrap_data.py:otomoto_scraping_flow
prefect deployment build -n otomoto-scraping -p default-agent-pool -q otomoto-scraping otomoto_scraping.py:otomoto_scraping_flow
python3 otomoto_scraping.py 
python3 otomoto_training.py 
python3 otomoto_scraping.py 
python3 otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
black otomoto_training.py
isort otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
isort otomoto_training.py 
black otomoto_training.py
isort otomoto_training.py 
black otomoto_training.py
isort otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
black otomoto_training.py
isort otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
isort otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
isort otomoto_training.py 
black otomoto_training.py
isort otomoto_training.py 
black otomoto_training.py
isort otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
isort otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
isort otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
isort otomoto_training.py 
python3 otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
isort otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
isort otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
isort otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
isort otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
ls
prefect server start
pipenv shell
/home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/python3.8
ls
cd orchestration/
ls -lah
deployment build -n mlops-gs-test -p default-agent-pool -q mlops-gs orchestrate_gs.py:main_flow_gs
prefect deployment build -n otomoto-price-prediction -p default-agent-pool -q tomoto-price-prediction orchestrate_training.py:main_flow
prefect deployment build -n otomoto-price-prediction -p default-agent-pool -q tomoto-price-prediction orchestrate_training.py:otomoto_training_flow
prefect deployment build -n otomoto-price-prediction -p default-agent-pool -q omoto-price-prediction orchestrate_training.py:otomoto_training_flow
prefect deployment build -n otomoto-price-prediction -p default-agent-pool -q omoto-price-prediction otomoto_training.py:otomoto_training_flow
python3 otomoto_training.py 
python3 lr.py
python3 otomoto_training.py 
prefect server stop
prefect server --help
python3 otomoto_training.py 
ls -lah
python3 otomoto_training.py 
ls
ls -lah
black otomoto_scrap_data.py 
prefect deployment build -n otomoto-scrap-data -p default-agent-pool -q otomoto-scrap-data otomoto_scrap_data.py:otomoto_scrap_data_flow
python3 otomoto_scrap_data.py 
ls -lah
git status
git add temp
git add orchestrate_s3.py 
git status
python3 otomoto_scrap_data.py 
cd
jupyter notebook
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
jupyter notebook
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
git add orchestration/otomoto_training.py
git add orchestration/config/config.json
git status
git add orchestration/otomoto_training_copy.py
git add .bash_history
git commit -m "otomoto_training changes"
git push
cd /home/konradballegro/orchestration/models
ls -lah
git status
cd
git status
git add orchestration/config/config.json orchestration/otomoto_training.py
git commit -m "otomoto_training changes"
git push
git status
git add notebooks/explainer_xgb.ipynb orchestration/models/xgb.model orchestration/otomoto_training.py
git commit -m "otomoto_training about right"
git push
git status
git add notebooks/outputs/reports/xgb_explainer.html notebooks/explainer_xgb.ipynb orchestration/models/xgb.model orchestration/otomoto_training.py
git commit -m "otomoto_training about right"
git push
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd orchestration/
python3 otomoto_scrap_data.py 
python3 otomoto_training.py 
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
black otomoto_training.py
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py 
isort otomoto_training.py
black otomoto_training.py
isort otomoto_training.py
black otomoto_training.py
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py 
black otomoto_training.py
python3 otomoto_training.py 
python3 otomoto_training.py
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
htop
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
ls
cd orchestration/
ls -lah
cd
prefect server start
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pipenv install --dev flake8
touch .flake8
pipenv uninstall flake8
git status
git add .bash_history .prefect/prefect.db notebooks/explainer_xgb.ipynb notebooks/explainer_xgb.ipynb notebooks/outputs/reports/xgb_explainer.html orchestration/otomoto_training.py 
git status
git commit -m "added docstrings to otomoto_training"
git push
git status
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd
pipenv shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd orchestration/
ls
python3 otomoto_training.py
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
flake8 otomoto_training.py
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
prefect server start
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
ls
prefect server start
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd orchestration/
ls
python3 otomoto_training.py
ls
isort otomoto_scraping.py 
black otomoto_scraping.py 
cd /home/konradballegro/orchestration/modules/scrapers
ls -lah
isort get_offers.py 
black get_offers.py 
isort offers_scraper.py 
black offers_scraper.py 
cd ..
ls -lah
python3 otomoto_scraping.py 
ls
python3 orchestrate.py
python3 orchestrate_gd.py
python3 orchestrate_gs.py
ls
python3 otomoto_training.py
cd /home/konradballegro/orchestration/mlruns
ls -lah
cd ..
ls
cd /home/konradballegro/orchestration
ls -lah
ls
python3 otomoto_training.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
git add .bash_history .prefect/prefect.db orchestration/data/inputs/manufacturers.txt orchestration/logs/app.log orchestration/models/xgb.model orchestration/modules/scrapers/get_offers.py orchestration/modules/scrapers/offers_scraper.py orchestration/otomoto_scraping.py orchestration/otomoto_training.py
git status
ls -lah
git status
git commit -m "added mlops tracking"
git push
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
/home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/python3.8
cd orchestration/
ls
python3 otomoto_training.py 
isort otomoto_training.py 
black otomoto_training.py 
python3 otomoto_training.py 
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
pipenv install streamlit
pip list > requirements.txt
ls -lah
streamli run /home/konradballegro/monitoring/streamlit_app/app.py
streamlit run /home/konradballegro/monitoring/streamlit_app/app.py
streamlit run /home/konradballegro/monitoring/app.py
ls
streamlit run streamlit_app/app.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
prefect server start
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd orchestration/
ls -lah
python3 otomoto_training.py
ls -lah
isort otomoto_training.py
python otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
git status
ls
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
isort otomoto_training.py
black otomoto_training.py
python3 otomoto_training.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd monitoring/
ls
cd /home/konradballegro/monitoring/config
isort config.py
black config.py
isort config.py
black config.py
isort config.py
black config.py
cd /home/konradballegro/monitoring
isort app.py
black app.py
cd
git status
git add .bash_history Pipfile monitoring/
git status
git add streamlit_app/
git add Pipfile_copy requirements_copy.txt orchestration/models/xgb.model
git status
git commit -m "monitoring changes"
git push
git status
pip list
cd /home/konradballegro/streamlit_app/utils
ls
isort ui.py
black ui.py
cd /home/konradballegro/streamlit_app
ls
isort app.py
black app.py
cd /home/konradballegro/monitoring
ls
isort app.py
black app.py
isort app.py
black app.py
ls
cd /home/konradballegro/streamlit_app
black app.py
cd /home/konradballegro/monitoring/config
black config.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
mkdir monitoring
cd monitoring
ls
mkdir config
mkdir data
mkdir fastapi
mkdir models
mkdir reports
mkdir src
mkdir static
mkdir streamlit_app
cd src
ls
mkdir scripts
mkd utils
mkdir utils
cd ..
ls
cd streamlit_app
touch app.py
ls
mkdir static
mkdir utils
cd utils
touch ui.py
cd ..
ls
isort app.py
black app.py
cd utils
ls
isort ui.py
black ui.py
ls
isort ui.py
black ui.py
isort ui.py
black ui.py
cd ..
ls
isort app.py
black app.py
isort app.py
black app.py
ls
streamlit run app.py
ls -lah
piplist
pip list
fastapi
xgboost
streamlit
ls
cd ..
ls
cd fastapi/
ls
isort app.py
black app.py
pip list
psql --help
sudo apt-get update
sudo apt-get install postgresql-client
psql --help
psql -h 0.0.0.0
psql -h 0.0.0.0 -U admin
psql -l
psql --help
psql --list
psql 
psql \?
psql -h
sudo apt-get install postgresql
sudo service postgresql status
sudo -i -u postgres
dpkg -l | grep postgresql
sudo systemctl restart postgresql
systemctl list-unit-files | grep postgres
cd
sudo apt-get remove --purge postgresql-*
sudo rm -rf /etc/postgresql/
sudo rm -rf /var/lib/postgresql/
sudo apt-get install postgresql
sudo systemctl status postgresql
dpkg -l | grep postgresql
sudo service postgresql status
sudo -i -u postgres
psql
sudo -i -u postgres
git status
git add .bash_history .prefect/prefect.db Pipfile requirements.txt orchestration/config/config.json 
git status
orchestration/data/green_tripdata_2021-01.parquet orchestration/data/green_tripdata_2021-02.parquet orchestration/data/outputs/abarth.csv orchestration/data/outputs/acura.csv orchestration/data/outputs/offers.csv 
orchestration/data/green_tripdata_2021-01.parquet orchestration/data/green_tripdata_2021-02.parquet orchestration/data/outputs/abarth.csv orchestration/data/outputs/acura.csv orchestration/data/outputs/offers.csv
git add orchestration/data/green_tripdata_2021-01.parquet orchestration/data/green_tripdata_2021-02.parquet orchestration/data/outputs/abarth.csv orchestration/data/outputs/acura.csv orchestration/data/outputs/offers.csv
git status
git add orchestration/otomoto_training.py orchestration/orchestrate_gs.py orchestration/orchestrate.py orchestration/modules/scrapers/offers_scraper.py orchestration/modules/scrapers/get_offers.py orchestration/models/xgb_copy.model
git status
git add orchestration/models/xgb.model orchestration/models/preprocessor.b orchestration/main_flow_gs-deployment.yaml orchestration/main_flow-deployment.yaml
git status
git commit -m "changes pre-monitoring"
git push
git status
git add monitoring .
git status
git reset HEAD~
git status
git add monitoring/
git status
git pull
git stash
git pull
git status
git add monitoring/
git status
git commit -m "monitoring init commit"
git push
git status
git add orchestration/config/config.json orchestration/models/xgb.model orchestration/otomoto_training.py 
git status
git add .gitignore monitoring/
git status
git commit -m "pre monitoring changes"
git push
cd /home/konradballegro/monitoring/src/utils
ls
isort data.py
black data.py
cd /home/konradballegro/monitoring/fastapi/app.py
black app.py
isort app.py
black app.py
cd /home/konradballegro/monitoring/fastapi
isort app.py
black app.py
/home/konradballegro/monitoring/fastapi
cd /home/konradballegro/monitoring/fastapi
ls
python3 app.py
ls
cd
ls
cd monitoring/
ls
python3 fastapi/app.py
cd
ls
python3 monitoring/fastapi/app.py
cd monitoring/
python3 fastapi/app.py
cd ..
python3 monitoring/fastapi/app.py
cd monitoring/
python3 mfastapi/app.py
python3 fastapi/app.py
cd /home/konradballegro/monitoring
ls
python3 app.py
pip list
pipenv shell
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pipenv install evidently
cd monitoring/
ls
python3 app.py
uvicorn
uvicorn app:app --reload
uvicorn app:app --log-level debug
ls
uvicorn app:app --log-level debug
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
git add monitoring/otomoto_monitoring.py orchestration/data/training/ temp/Pipfile_copy temp/requirements_copy.txt web-app/
git status
git reset HEAD~
git status
git pull
git stash
git pull
git pull force
git pull
git status
git add mlops-zoomcamp monitoring/app.py scoring-batch/score.py scoring-batch/score_backfill.py scoring-batch/score_deploy.py
git commit -m "deleted some"
git push
git status
git add monitoring/otomoto_monitoring.py scoring-event/ temp/Pipfile_copy temp/requirements_copy.txt
git status
git add .gitignore 
git status
git commit -m "added scoring-event"
git push
git status
cd
mkdir scoring-batch
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd web-scoring/
ls -lag
ls -lah
python3 test.py
python3 otomoto_scoring_test.py
python3 test.py
python3 otomoto_scoring_test.py
python3 test.py
python3 predict.py
python3 test.py
python3 predict.py
python3 otomoto_scoring_test.py
python3 test.py
python3 otomoto_scoring_test.py
ls
isort otomoto_scoring.py
isort otomoto_scoring_test.py
black otomoto_scoring.py
black otomoto_scoring_test.py
python3 otomoto_scoring_test.py
black otomoto_scoring.py
black otomoto_scoring_test.py
python3 otomoto_scoring_test.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd /home/konradballegro/web-scoring
ls -lah
flask run otomoto_scoring.py
ls -lah
python3 otomoto_scoring.py
ls -lah
python3 predict.py
python3 otomoto_scoring.py
python3 predict.py
python3 otomoto_scoring.py
python3 predict.py
python3 otomoto_scoring.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd scoring-event/
ls -lah
python3 predict.py 
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd monitoring/
uvicorn app:app
ls
uvicorn otomoto_monitoring:app
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 monitoring/app.py
streamlit run streamlit_app/app.py
streamlit run web-app/otomoto_app.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
ls
mkdir scraping
ls -lah
python3 training/otomoto_training.py 
python3 scraping/otomoto_scraping.py 
python3 training/otomoto_training.py 
isort scoring-event/app.py 
black scoring-event/app.py 
black scoring-event/config/config.json
python3 training/otomoto_training.py 
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd monitoring/
ls -lah
uvicorn otomoto_monitoring:app
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
cd scoring-event/
ls -lah
python3 otomoto_scoring.py
python3app.py
python3 app.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
git staus
git status
git add models/
git add streamlit/
git add scraping/
git add training/
git status
git add .bash_history .gitignore monitoring/reports/model_performance.html monitoring/reports/target_drift.html orchestration/
git status
git add scoring-event/
git status
git addd requirements_copy.txt scoring-event/scoring.log streamlit_app/
git add requirements_copy.txt scoring-event/scoring.log streamlit_app/
git status
git add web-service 
git status
git commit -m "added scoring of single evnet usinf Flask; projects restructure"
git push
git status
git push
git rm scoring-event/app.log
git push
git reset HEAD~
git status
git add models/ scoring-event/
git status
git add web-service/
git add orchestration/
git status
git add streamlit_app/
git add monitoring/
git status
git add requirements_copy.txt 
git add .gitignore 
git status
git commit -m "projects restructure"
git push
git status
git add scraping/
git add streamlit/
git add training/
git commit -m "added scoring of an event"
git push
python3 scoring-event/otomoto_scoring_event.py 
black scoring-event/config.json
black scoring-event/config/config.json
black scoring-event/app.py
isort scoring-event/app.py
black scoring-event/app.py
python3 scoring-event/otomoto_scoring_event.py 
git status
git add .bash_history models/xgb.model scoring-event/app.py scoring-event/otomoto_scoring_event.py training/otomoto_training.py scoring-event/config/ scoring-event/test_data_preprocess.csv scoring-event/test_features_engineer.csv
git status
git commit -m "initial scoring-event"
git push
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd scoring-event/
ls
python3 otomoto_scoring_test.py
python3 otomoto_scoring_event.py
cd
python3 scoring-event/app.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd  model
cd  models
ls -lah
cd
cd streamlit/
ls -lag
streamlit run app/app.py
streamlit run app:app.py
streamlit run app/app.py
streamlit run app.py
cd
streamlit run streamlist/app.py
streamlit run streamlit/app.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
prefect server start
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
/home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/python3.8
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd
git status
git add .bash_history 
git add models/xgb.model 
git add monitoring/
git status
git add data/inputs/header_en.txt
git add data/inputs/header_en.txt -f
git status
git add data/inputs/header_pl.txt -f
git add data/inputs/iveco.txt -f
git add data/inputs/offers.txt -f
git add data/inputs/manufacturers_batch.txt -f
git add data/inputs/manufacturers.txt -f
git add data/inputs/offers.txt -f
git add data/inputs/offers.csv -f
git add data/inputs/iveco.csv -f
git status
git add scoring-event 
git add
git status
git add scoring-batch/
git status
git commit -m "added scoring-batch module"
git push
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 scoring-event/otomoto_scoring_event.py 
isort scoring-event/otomoto_scoring_event.py 
black scoring-event/otomoto_scoring_event.py 
isort scoring-event/otomoto_scoring_event.py 
black scoring-event/otomoto_scoring_event.py 
python3 scoring-event/otomoto_scoring_event.py 
isort scoring-event/otomoto_scoring_event.py 
python3 scoring-event/otomoto_scoring_event.py 
black scoring-event/otomoto_scoring_event.py 
python3 scoring-event/otomoto_scoring_event.py 
isort scoring-event/otomoto_scoring_event.py 
black scoring-event/otomoto_scoring_event.py 
python3 scoring-event/otomoto_scoring_event.py 
isort scoring-event/otomoto_scoring_event.py 
black scoring-event/otomoto_scoring_event.py 
python3 scoring-event/otomoto_scoring_event.py 
isort scoring-event/otomoto_scoring_event.py 
python3 scoring-event/otomoto_scoring_event.py 
black scoring-event/otomoto_scoring_event.py 
python3 scoring-event/otomoto_scoring_event.py 
isort scoring-event/otomoto_scoring_event.py 
black scoring-event/otomoto_scoring_event.py 
python3 scoring-event/otomoto_scoring_event.py 
black scoring-event/otomoto_scoring_event.py 
python3 scoring-event/otomoto_scoring_event.py 
ls -lah
python3 scoring-event/otomoto_scoring_event.py 
cd /home/konradballegro/data/inputs
ls -lah
cd
python3 scoring-event/otomoto_scoring_event.py 
cd /home/konradballegro/data/inputs
ls -lah
cd
python3 scoring-event/otomoto_scoring_event.py 
python3 scoring-event/otomoto_scoring_batch.py 
python3 scoring-batch/otomoto_scoring_batch.py 
isort scoring-batch/otomoto_scoring_batch.py 
black scoring-batch/otomoto_scoring_batch.py 
isort scoring-batch/otomoto_scoring_batch.py 
black scoring-batch/otomoto_scoring_batch.py 
python3 scoring-batch/otomoto_scoring_batch.py 
python3 scoring-batch/app.py 
python3 scoring-batch/otomoto_scoring_batch.py 
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 scoring-event/app.py
python3 scoring-batch/app.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd monitoring/
ls -lah
uvicorn otomoto_monitoring.py:app
cd
ls
streamlit run streamlit:app.py
cd streamlit/
streamlit run streamlit:app.py
cd
cd 
streamlit run app:app
streamlit run app:app.py
cd streamlit/
streamlit run app:app.py
ls -lah
streamlit run app.py
cd
streamlit run streamlit:app.py
ls
streamlit run streamlit:app.py
cd streamlit/
streamlit run app.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd monitoring/
ls -lah
uvicorn otomoto_monitoring.py 
uvicorn otomoto_monitoring:app
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 training/otomoto_training.py 
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
prefect server start
pipenv shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
ls -lah
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
ls -lah
python3 scoring-batch/app.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 scoring-batch/otomoto_scoring_event.py 
python3 scoring-batch/otomoto_scoring_batch.py 
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 training/otomoto_training.py 
cd
git status
git add .bash_history
git add data/inputs/iveco.csv 
git add data/inputs/offers.csv 
git add models/xgb.model 
git add scoring-batch/app.py
git add scoring-batch/otomoto_scoring_batch.py 
git commit -m "improved otomotomo_scoring.batch.py"
git push
git status
python3 training/otomoto_training.py 
git status
git add models/xgb.model 
git add monitoring/data/current/offers_current.csv 
git add monitoring/data/reference/y_pred.csv 
git status
git add training/config/config.json 
git add training/otomoto_training.py 
git commit -m "improved hyperparameters_gridsearch"\
git commit
git push
git status
python3 training/otomoto_training.py 
git status
git add models/xgb.model 
git add training/otomoto_training.py 
git commit -m "improved otomoto_training.py"
git push
python3 training/otomoto_training.py 
git status
git add models/xgb.model 
git add monitoring/data/reference/offers_reference.csv 
git add scoring-batch/app.py
git add scoring-batch/config/config.json 
git add scoring-batch/test_features_engineer.csv 
git add training/otomoto_training.py 
git status
git add data/inputs/offers.csv 
git commit -m "improved otomoto_scoring_batch"
git push
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
prefect server start
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 scoring-batch/app.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
prefect server start
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
prefect server start
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 scoring-batch/otomoto_scoring_batch.py 
python3 scoring-batch/otomoto_scoring_batch.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 /home/konradballegro/scraping/otomoto_scraping.py
python3 /home/konradballegro/scoring-batch/otomoto_scoring_batch.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 /home/konradballegro/scraping/otomoto_scraping.py
python3 /home/konradballegro/training/otomoto_training.py
git status
python3 /home/konradballegro/training/otomoto_training.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 /home/konradballegro/training/otomoto_training.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
git add .bash_history 
git add data/inputs/
git status
git add monitoring/config/config.py 
git add scoring-batch/config/config.json 
git add training/config/config.json 
git add training/otomoto_training.py 
git status
git add data/raw/abarth.csv 
git status
git add data/raw/iveco.csv 
git add data/raw/offers.csv 
git status
git add scraping/otomoto_scraping.py
git add scraping/scrapers/
git status
git add monitoring/reports/
git status
git add monitoring/src/utils/data.py
git status
git commit -m "restructure of a project"
git push
git status
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
git status
git push
git status
git add data/raw/
git status
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd streamlit/
streamlit run app.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd monitoring
uvicorn otomoto_monitoring:app
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 /home/konradballegro/scoring-batch/otomoto_scoring_batch.py
git status
git add data/metadata/manufacturers_batch.txt
git add data/raw/abarth.csv
git add data/raw/iveco.csv
git add data/raw/offers.csv
git add scraping/scrapers/offers_scraper.py
git add data/raw/
git status
git commit -m "added appending for scoring-batch"
git push
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
python3 scoring-batch/app.py 
pipenv shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 scoring-batch/otomoto_scoring_batch.py 
git status
git add .bash_history 
git add data/raw/offers.csv
git add scoring-batch/
git status
fit add data/raw/
git add data/raw/
git add data/scored/
git add data/preprocessed/
git status
git commit -m "added saving of scored-batch"
git push
git status
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
ping 192.168.0.241
cd
python3 scoring-batch/app.py 
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
cd streamlit/
streamlit run app.py
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
htop
pipenv shell
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 /home/konradballegro/scoring-batch/otomoto_scoring_batch.py
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
python3 scoring-batch/app.py 
source /home/konradballegro/.virtualenvs/konradballegro-pfUEMlPh/bin/activate
pipenv shell
python3 scoring_batch/otomoto_scoring_batch.py
python3 tests/model_test.py 
python3 scoring_batch/otomoto_scoring_batch.py
cd
git status
git add .bash_history
git add data/raw/
git status
git add Pipfile 
git add data/preprocessed/
git add monitoring/
git status
git add scoring-batch 
git status
git add training/
git status
git add scoring_batch/
git add scraping/
git add tests/
git status
git add scoring_batch/app.log
git status
git add scoring_batch/app.log
git status
git commit -m "added unit tests"
git push
git status
python3 scoring_batch/app.py
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
pipenv install dev pytest
pipenv install pytest --dev
prefect server start
