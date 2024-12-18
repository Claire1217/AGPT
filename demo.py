from PIL import Image

from Args import build_args
from models.mdetr import * 
from models.transvg import *
from dataloader.mscxr_dataloader import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def prepare_input(args, img_fn):
    # Load and preprocess the image
    image = Image.open(img_fn).convert("RGB")
    image = np.array(image)
    transform = get_transforms(args, 'val', False)
    transformed = transform(image=np.array(image))
    image_tensor = transformed['image'].unsqueeze(0).to(args.device)  # Add batch dimension
    image_mask = torch.zeros_like(image_tensor[:, 0, :, :], dtype=torch.bool)
    image_nested = NestedTensor(image_tensor, image_mask)
    return image_nested

def get_pred(args, model, img_fn):
    image_nested = prepare_input(args, img_fn)
    phrase, _ = examples[img_fn]
    with torch.no_grad():
        outputs = model(image_nested, [phrase])
    pred_boxes = outputs['pred_boxes'].squeeze().view(-1, 4).detach().cpu()
    pred_boxes = xywh2xyxy(pred_boxes)
    return pred_boxes

def save_plot(args, img_fn, pred_box, gt_box, save_dir):
    """
    Visualize bounding boxes on the image and save the result to the output folder.
    """
    image = Image.open(img_fn).convert("RGB")
    plt.imshow(image)
    if gt_box is not None:
        x1, y1, w, h = gt_box 
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    x1, y1, x2, y2 = pred_box.squeeze() * args.imsize
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none')
    plt.gca().add_patch(rect)
    fn = os.path.join(save_dir, os.path.basename(img_fn))
    plt.savefig(fn)
    plt.close()
    return

def inference(model_name):
    save_dir = os.path.join('demo/outputs', model_name)
    os.makedirs(save_dir, exist_ok=True)  
    # Load model
    if model_name == 'transvg':
        args = build_args('transvg', 'ms_cxr')
        model = build_transvg_model(args)
        transvg_ckp = torch.load('model_weight/transvg.pth', map_location='cpu')
        model.load_state_dict(transvg_ckp['model'], strict=False)
    elif model_name == 'mdetr':
        args =  build_args('mdetr', 'ms_cxr')
        model = build_mdetr_model(args)
        mdetr_ckp = torch.load('model_weight/mdetr.pth', map_location='cpu')
        model.load_state_dict(mdetr_ckp['model'], strict=False)
    model = model.to(args.device)
    # Save plots
    model.eval()
    for img_fn in examples.keys():
        phrase, gt_box = examples[img_fn]
        pred_box = get_pred(args, model, img_fn)
        save_plot(args, img_fn, pred_box, gt_box, save_dir)
    return
   

examples = {
    'demo/0ae1a7ad-fcc627aa-01e96e50-cb14cf17-23f441ac.jpg': ('small left apical pneumothorax', [345, 70, 141, 45]),
    'demo/3aa9ee81-08355b5d-713a3a03-6cd20281-6222326c.jpg': ('Left perihilar opacity', [400, 274, 94, 140]),
    'demo/8350f725-7918f8aa-abc95fec-a3d62a15-a47db0d9.jpg': ('patchy consolidation in the right lower', [60, 293, 146, 216]),
    'demo/f7b69ee3-db7f264c-fca7d1c7-1d372fc0-02b35a47.jpg': ('there is mild cardiomegaly' , [195, 281, 338, 198]),
}


if __name__ == '__main__':
    inference('transvg')
    inference('mdetr')