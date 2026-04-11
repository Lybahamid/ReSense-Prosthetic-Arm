# ReSense-Prosthetic-Arm
An ML enabled Prosthetic Limb

# Steps to run
venv\Scripts\activate
python -m Src.data_loader
python -m Src.preprocessing
python -m Src.preprocessing
python -m Src.feature_extraction
python -m Src.SVM
python -m Src.traincnn
python train_hybrid_ensemble.py
python Train_TL.py(Ensemble)
python TL( No Ensemble)
(for testing) python test_ensemble.py
(for testing)python simulate_realtime.py
(for testing)python Simulation.py
