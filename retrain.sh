export PYTHONPATH=/Users/maciejsaczewski/Documents/Inzynier/Object_detection_Coral/slim:$PYTHONPATH

NUM_TRAINING_STEPS=50 && \
NUM_EVAL_STEPS=10

./scripts/retrain_detection_model.sh \
--num_training_steps ${NUM_TRAINING_STEPS} \
--num_eval_steps ${NUM_EVAL_STEPS}
