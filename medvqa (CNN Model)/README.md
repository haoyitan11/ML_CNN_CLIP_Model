CNN Model
cd "D:\UM\Document\WOA7001 ADVANCED MACHINE LEARNING\Alternative assessment\Coding\Coding (CNN and CLIP Model)\medvqa (CNN Model)"

- Install Dependencies
python -m pip install -r requirements.txt
pip install torchxrayvision
python -m pip install -e .

- 1.Build answer vocabulary
python scripts\build_vocab.py --dataset_name flaviagiammarino/vqa-rad --top_k 300 --out_path outputs\answer_vocab.json

- 2.Train CNN baseline
python scripts\train_cnn.py --dataset_name flaviagiammarino/vqa-rad --vocab_path outputs\answer_vocab.json --exp_dir outputs\exp_cnn --epochs 4

- 3.Evaluate using best checkpoint
python scripts\eval_cnn.py --dataset_name flaviagiammarino/vqa-rad --vocab_path outputs\answer_vocab.json --ckpt_path outputs\exp_cnn\checkpoints\best.pt --exp_dir outputs\exp_cnn

- 4. Run Gradio Studiom
python app_gradio.py

- 5. Extract images from HuggingFace
python scripts\03_export_images.py --dataset_name flaviagiammarino/vqa-rad --out_dir "D:\UM\Document\WOA7001 ADVANCED MACHINE LEARNING\Alternative assessment\Coding\Coding (CNN and CLIP Model)\medvqa (CNN Model)\images" --save_meta
