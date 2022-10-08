echo [$(date)]:"init_setup .sh file initiated"
echo [$(date)]: "Stated creating conda environment"
conda create --prefix ./enve_deep python=3.8 -y
echo [$(date)]: "Activating environment"
source activate ./enve_deep
echo [$(date)]: "insatlling required packages"
pip install -r requirements_dev.txt