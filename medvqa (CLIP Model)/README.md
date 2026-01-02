CILP Model
Upgrade transformers
pip install --upgrade transformers torch

cd "D:\UM\Document\WOA7001 ADVANCED MACHINE LEARNING\Alternative assessment\Coding\Coding (CNN and CLIP Model)\medvqa (CLIP Model)"

1. Install Dependencies
pip install -r requirements.txt

2. Setup dataset
python scripts\export_images.py --dataset_name flaviagiammarino/vqa-rad --out_dir images --save_meta

3. Build Vocabulary
python scripts\build_vocab.py --out_path outputs\vocabs.json

4. Train the CILP model
python scripts\train_clip.py --dataset_name flaviagiammarino/vqa-rad --vocab_bundle outputs\vocabs.json --exp_dir outputs\exp_clip --epochs 4
	
5. Evaluate the CILP model
python scripts\eval_clip.py --dataset_name flaviagiammarino/vqa-rad --vocab_bundle outputs\vocabs.json --ckpt_path outputs\exp_clip\checkpoints\best.pt --exp_dir outputs\exp_clip

6. Run the Gradio Client
python scripts\app_gradio.py --vocab_bundle outputs\vocabs.json --ckpt_path outputs\exp_clip\checkpoints\best.pt