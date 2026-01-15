CNN Model
cd "D:\UM\Document\WOA7001 ADVANCED MACHINE LEARNING\Alternative assessment\Coding\Coding (CNN and CLIP Model)\medvqa (CNN Model)"

- Install Dependencies
python -m pip install -r requirements.txt
pip install torchxrayvision
python -m pip install -e .

- 1.Build answer vocabulary
python scripts\build_vocab.py --dataset_name flaviagiammarino/vqa-rad --top_k 300 --out_path outputs\answer_vocab.json

- 2.Train CNN baseline
python scripts\train_cnn.py --dataset_name flaviagiammarino/vqa-rad --vocab_path outputs\answer_vocab.json --exp_dir outputs\exp_cnn --epochs 40

- 3.Evaluate using best checkpoint
python scripts\eval_cnn.py --dataset_name flaviagiammarino/vqa-rad --vocab_path outputs\answer_vocab.json --ckpt_path outputs\exp_cnn\checkpoints\best.pt --exp_dir outputs\exp_cnn

- 4. Run Gradio Studio
python app_gradio.py 

- 5. Run both
python app_gradio.py --startup_model both 
python both_model_app.py --startup_model cnn
python both_model_app.py --startup_model clip

- 6.train_loss, train_accuracy
python scripts\loss_accuracy.py

Both model
python .\scripts\both_model_loss_accuracy.py `
  --cnn_history "D:\UM\Document\WOA7001 ADVANCED MACHINE LEARNING\Alternative assessment\Coding\Coding (CNN and CLIP Model)\medvqa (CNN Model)\outputs\exp_cnn\history.json" `
  --clip_history "D:\UM\Document\WOA7001 ADVANCED MACHINE LEARNING\Alternative assessment\Coding\Coding (CNN and CLIP Model)\medvqa (CLIP Model)\outputs\exp_clip\history.json"

- 7. Extract images from HuggingFace
python scripts\03_export_images.py --dataset_name flaviagiammarino/vqa-rad --out_dir "D:\UM\Document\WOA7001 ADVANCED MACHINE LEARNING\Alternative assessment\Coding\Coding (CNN and CLIP Model)\medvqa (CNN Model)\images" --save_meta
