sudo apt-get update
sudo apt install python3.8-venv
python3 -m venv mlops-zoomcamp
source mlops-zoomcamp/bin/activate
pip install -U pip
pipenv shell
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
rm -rf mlops-zoomcamp
pip list
apt install pipenc
apt install pipenv
pipenv shell
sudo apt install pipenv
pipenv shell
pip shell
sudo pip shell
source .virtualenvs/konradballegro-pfUEMlPh/bin/activate
pipenv shell
gcloud compute firewall-rules create mlflow-tracking-server     --network default     --priority 1000     --direction ingress     --action allow     --target-tags mlflow-tracking-server     --source-ranges 0.0.0.0/0     --rules tcp:5000 \
clear
gcloud compute firewall-rules create mlflow-tracking-server-1     --network default     --priority 1000     --direction ingress     --action allow     --target-tags mlflow-tracking-server-1     --source-ranges 0.0.0.0/0     --rules tcp:5000     --enable-logging
ls
source .virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
touch lr.py
vim lr.py
cat lr.py
python3 lr.py
pipenv shell
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pipenv install prefect
pip list
prefect server start
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pipenv install mlflow scikit-learn
pip list
ls
python3 lr.py
pipenv install google-cloud-storage
pip list
python3 lr.py
source .virtualenvs/konradballegro-pfUEMlPh/bin/activate
git commit -m "first commit"
git config --global user.mail "borowiec.k@gmail.com"
git config --global user.name "Konrad Borowiec"
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/KonuTech/mlops-zoomcamp-project.git
git push -u origin main
source .virtualenvs/konradballegro-pfUEMlPh/bin/activate
prefect server start
python3 lr.py
pip list
ls -lah
echo "# mlops-zoomcamp-project" >> README.md
git init
touch .gitignore
vim .gitignore
cat .gitignore
vim .gitignore
cat .gitignore
vim .gitignore
git add README.md
git add .gitignore
git add .prefect
git add .bash_history
git add .bashrc
git add Pipfile
git add lr.py
git status
git add .gitignore
pipenv shell
. /home/konradballegro/.local/share/virtualenvs/konradballegro-pfUEMlPh/bin/activate
pip list
git clone https://github.com/DataTalksClub/mlops-zoomcamp.git
ls -lah
cd /home/konradballegro/mlops-zoomcamp/03-orchestration
ls
ls -lah
mkdir data
python3 orchestrate.py
pipenv install xgboost
pip list
python3 orchestrate.py
prefect deploy orchestrate.py:main_flow -n taxi_local_data
prefect deployment orchestrate.py:main_flow -n taxi_local_data
prefect deploy orchestrate.py:main_flow -n taxi_local_data
pip list
prefect deploy orchestrate.py:main_flow -n taxi_local_data
prefect --help
prefect worker start -p zoompool
prefect work-pool start -p zoompool
pip list
prefect --version
prefect deployment build -n mlops-test -p default-agent-pool -q mlops orchestrate.py:main_flow
pipenv install -U xgboost
pipenv install xgboost
pip install xgboost
pip list
prefect deployment build -n mlops-test -p default-agent-pool -q mlops orchestrate.py:main_flow
python3 orchestrate.py
git add mlops-zoomcamp
cd ..
git add mlops-zoomcamp
ls
cd
ls
git add mlops-zoomcamp/
git rm --cached mlops-zoomcamp/
git status
git add mlops-zoomcamp/
git status
git add .bash_history 
git add .prefect/
git status
git add mlops-zoomcamp
git commit -m "second commit"
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
