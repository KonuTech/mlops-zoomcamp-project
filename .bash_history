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
