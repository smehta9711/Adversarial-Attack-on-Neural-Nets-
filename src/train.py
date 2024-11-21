import torch
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
import time
import os
from models.adversarial_models import AdversarialModels
from utils.dataloader import LoadFromImageFile
from utils.utils import makedirs, to_cuda_vars, format_time
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
from torch.optim.lr_scheduler import StepLR


parser = argparse.ArgumentParser(description='Generating Adversarial Patches')
parser.add_argument('--data_root', type=str, help='path to dataset', default='Src/input_img')
parser.add_argument('--train_list', type=str, default='Src/list/test_list.txt')
parser.add_argument('--print_file', type=str, default='Src/list/printable30values.txt')
parser.add_argument('--distill_ckpt', type=str, default="models/guo/distill_model.ckpt")
parser.add_argument('--height', type=int, help='input image height', default=256)
parser.add_argument('--width', type=int, help='input image width', default=512)
parser.add_argument('-b', '--batch_size', type=int, help='mini-batch size', default=32)
parser.add_argument('-j', '--num_threads', type=int, help='data loading workers', default=4)
parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--num_epochs', type=int, help='number of total epochs', default=50)
parser.add_argument('--seed', type=int, help='seed for random functions, and network initialization', default=0)
parser.add_argument('--patch_size', type=int, help='Resolution of patch', default=64)
parser.add_argument('--patch_shape', type=str, help='circle or square', default='circle')
parser.add_argument('--patch_path', type=str, help='Initialize patch from file')
parser.add_argument('--mask_path', type=str, help='Initialize mask from file')
parser.add_argument('--target_disp', type=int, default=70)
parser.add_argument('--model', nargs='*', type=str, default='distill', choices=['distill'], help='Model architecture')
parser.add_argument('--name', type=str, help='output directory', default="result")
args = parser.parse_args()

# Loss functions
def depth_loss(pred_disp, target_disp, mask):
    masked_loss = torch.mean(torch.abs(pred_disp[mask == 1] - target_disp[mask == 1]))
    return masked_loss 

def nps_loss(patch_batch, printable_colors):
    # Reshape the patch_batch to (batch_size, height * width, channels)
    patch_flat = patch_batch.permute(0, 2, 3, 1).reshape(-1, 3)
    
    # Compute pairwise distances between each pixel in the patch and the printable colors
    distances = torch.cdist(patch_flat.unsqueeze(0), printable_colors.unsqueeze(0)).squeeze(0)
    
    # Compute the minimum distance to any printable color for each pixel
    min_distances, _ = torch.min(distances, dim=1)
    
    # Sum the minimum distances for all pixels
    return torch.sum(min_distances)

def total_variation_loss(patch_batch):
    # Compute differences between adjacent pixels along height and width dimensions
    diff_h = torch.abs(patch_batch[:, :, 1:, :] - patch_batch[:, :, :-1, :])  # Difference in height
    diff_w = torch.abs(patch_batch[:, :, :, 1:] - patch_batch[:, :, :, :-1])  # Difference in width
    
    # Sum the differences for total variation loss
    return torch.sum(diff_h) + torch.sum(diff_w)
# Function to load printable colors
def load_printable_colors(print_file):
    colors = []
    with open(print_file, 'r') as f:
        for line in f:
            rgb = line.strip().split(',')
            colors.append([float(c) for c in rgb])
    return torch.tensor(colors, device='cuda:1')

# Function to create a center mask for each image in the batch
def create_center_mask(image_shape, patch_shape):
    """
    Create a binary mask with the same size as the image, where the patch
    will be placed in the center. The rest of the mask is filled with zeros.
    Handles batches of images.
    """
    batch_size, channels, img_h, img_w = image_shape
    _, _, patch_h, patch_w = patch_shape

    # Initialize mask with zeros
    mask = torch.zeros((batch_size, channels, img_h, img_w), dtype=torch.float32)

    # Get the starting and ending coordinates to place the patch at the center
    c_y, c_x = img_h // 2, img_w // 2
    y1, y2 = c_y - patch_h // 2, c_y + patch_h // 2
    x1, x2 = c_x - patch_w // 2, c_x + patch_w // 2

    # Apply the patch in the center of each image in the batch
    mask[:, :, y1:y2, x1:x2] = 1

    return mask

# Function to apply the patch using the mask for each image in the batch
def apply_patch_to_image_batch(image, patch, mask):
    """
    Apply the patch to the image using the mask for a batch of images.
    (1 - mask) * image + mask * patch
    The patch will be placed in the masked region for each image in the batch.
    """
    batch_size = image.shape[0]

    # Broadcast the patch to be applied across the batch
    # patch_expanded = patch.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Apply the patch to the image using the mask
    patched_image = (1 - mask) * image + mask * patch
    return patched_image

def save_loss_plot(disp_loss_list, nps_loss_list, tv_loss_list, combined_loss_list, save_path):
    plt.figure(figsize=(10, 6))
    
    # Plot each loss
    plt.plot(disp_loss_list, label="Disparity Loss (L_depth)", color="blue")
    plt.plot(nps_loss_list, label="NPS Loss", color="green")
    plt.plot(tv_loss_list, label="Total Variation Loss", color="red")
    plt.plot(combined_loss_list, label="Combined Loss", color="orange")
    
    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend(loc='upper right')

    # Save the plot
    plt.savefig(os.path.join(save_path, "70_disp_loss.png"))
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    save_path = 'Dst/checkpoints/' + args.name
    print('===============================')
    print('=> Everything will be saved to \"{}\"'.format(save_path))
    makedirs(save_path)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Create the training transformations pipeline for input images
    train_transform = T.Compose([
        T.ToPILImage(), 
        T.Resize((args.height, args.width)),  
        T.RandomPerspective(distortion_scale=0.5, p=0.5),
        T.RandomAffine(degrees=30, scale=(0.5, 1.5)),  
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomHorizontalFlip(p=0.5),                    
        T.ToTensor()                                      
    ])

    # Initialize dataset
    train_set = LoadFromImageFile(
        args.data_root,
        args.train_list,
        seed=args.seed,
        train=True,
        monocular=True,
        transform=train_transform,
        extension=".png"
    )

    # Create a data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_threads,
        pin_memory=True,
        drop_last=True
    )

    print('===============================')
    # Attacked Models
    models = AdversarialModels(args)
    models.load_ckpt()

    # Patch initialization (shared for the entire batch)
    patch_cpu = torch.rand((3, args.patch_size, args.patch_size), requires_grad=True)

    # Optimizer
    optimizer = torch.optim.Adam([patch_cpu], lr=args.lr, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # Load printable colors
    printable_colors = load_printable_colors(args.print_file)

    # Training loop
    print('===============================')
    print("Start training ...")
    start_time = time.time()
    
    disp_loss_list, nps_loss_list, tv_loss_list, combined_loss_list = [], [], [], []


    for epoch in range(args.num_epochs):
        ep_nps_loss, ep_tv_loss, ep_loss, ep_disp_loss = 0, 0, 0, 0
        ep_time = time.time()

        for i_batch, sample in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=len(train_loader), leave=False):
            optimizer.zero_grad()

            # Move sample to device
            sample = to_cuda_vars(sample)
            sample.update(models.get_original_disp(sample))  # Get original disparity map

            img, original_disp = sample['left'], sample['original_distill_disp']
            
            patch_t = train_transform(patch_cpu)
            patch_batch = patch_t.unsqueeze(0).expand(args.batch_size, -1, -1, -1).cuda()
            
            

            # Create a mask for each image in the batch
            mask_batch = create_center_mask(img.shape, patch_batch.shape).to(device)

            # Apply transformations to the patch (done once for the batch)
            

            # Apply the patch to the image batch using the mask
#             patched_image = apply_patch_to_image_batch(img, patch_t, mask_cpu)
#             est_disp = models.distill(patched_image)
            
            # Expand the patch to have the batch size dimension so it can be applied to the entire batch
            
            # mask_batch = mask_cpu.unsqueeze(0).expand(batch_size, -1, -1, -1)

            # Apply the patch to the image batch using the mask
            patched_image = apply_patch_to_image_batch(img, patch_batch, mask_batch).cuda()
            
            # with torch.no_grad():
            est_disp = models.distill(patched_image).cuda()

            # Target disparity for patch area
            target_disp = original_disp.clone()
            target_disp[mask_batch[:,0:1,:,:] == 1] = args.target_disp
            
            # print('Est and target disp shapes ', est_disp.shape, target_disp.shape)


            # # Target disparity for patch area
            # target_disp = original_disp.clone()
            # target_disp[mask_cpu == 1] = args.target_disp

            # Losses
            disp_loss = depth_loss(est_disp, target_disp, mask_batch[:,0:1,:,:].cuda())
            nps_loss_value = nps_loss(patch_batch.cuda(), printable_colors.cuda())
            tv_loss_value = total_variation_loss(patch_batch.cuda())


            # Combine losses
            alpha, beta = 0.00001, 0.00001
            loss = 0.1*disp_loss + alpha * nps_loss_value + beta * tv_loss_value

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Ensure patch values stay within valid image range [0, 1]
            patch_cpu.data.clamp_(0, 1)

            # Log losses
            ep_disp_loss += disp_loss.item()
            ep_nps_loss += nps_loss_value.item()
            ep_tv_loss += tv_loss_value.item()
            ep_loss += loss.item()

        # Learning rate scheduler step
        scheduler.step(ep_loss / len(train_loader))
        
        # Append losses to the lists for plotting
        disp_loss_list.append(ep_disp_loss)
        nps_loss_list.append(ep_nps_loss)
        tv_loss_list.append(ep_tv_loss)
        combined_loss_list.append(ep_loss)


        ep_time = time.time() - ep_time
        total_time = time.time() - start_time
        print('===============================')
        print(f'FINISHED EPOCH {epoch}')
        print(f'TOTAL TIME: {format_time(int(total_time))}')
        print(f'EPOCH TIME: {format_time(int(ep_time))}')
        print(f'EPOCH LOSS: {ep_loss / len(train_loader)}')
        print(f'DISP LOSS: {0.1*ep_disp_loss / len(train_loader)}')
        print(f'NPS LOSS: {0.00001 * ep_nps_loss / len(train_loader)}')
        print(f'TV LOSS: {0.00001 * ep_tv_loss / len(train_loader)}')

        # # Save patch and mask
        # torch.save(patch_cpu.data, os.path.join(save_path, f'epoch_{epoch}_patch.pt'))
        # torch.save(mask_cpu.data, os.path.join(save_path, f'epoch_{epoch}_mask.pt'))
        
        np.save(save_path + '/epoch_{}_70disp_patch.npy'.format(str(epoch)), patch_cpu.data.numpy())
        np.save(save_path + '/epoch_{}_70disp_mask.npy'.format(str(epoch)), mask_batch[0,:,:,:].cpu().data.numpy())
    
    save_loss_plot(disp_loss_list, nps_loss_list, tv_loss_list, combined_loss_list, save_path)


if __name__ == '__main__':
    main()
