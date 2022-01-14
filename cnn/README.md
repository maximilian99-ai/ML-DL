# Divide dataset

    python divide_dataset.py

# Train cnn

    python main_cnn.py --data ./data -b 64 --lr 0.01 --num-classes 5

# Validate cnn

    python main_cnn.py -e --data ./data -b 64 --num-classes 5 --resume model_best.pth

# Test cnn

    python main_cnn_test.py --data ./data/test --num-classes 5 --resume model_best.pth
