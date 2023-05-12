"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from pytorch_fid import fid_score
from torchvision.transforms import Resize
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    fid_scores = []                # list to save FID scores

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        #model.update_learning_rate()    # update learning rates in the beginning of every epoch. taken out to fix bug
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # ADDED: Update learning rates after calling optimize_parameters

            # Call the update_learning_rate() function after the optimizer steps
            optimizer_G = model.optimizer_G
            optimizer_D = model.optimizer_D
            update_lr = (i == len(dataset) - 1)
            print_lr = update_lr and (epoch == 1 or epoch % opt.save_epoch_freq == 0)
            model.update_learning_rate([optimizer_G, optimizer_D], update_lr=update_lr, print_lr=print_lr)
            # optimizer_G = model.optimizer_G
            # optimizer_D = model.optimizer_D
            # model.update_learning_rate([optimizer_G, optimizer_D], print_lr=(epoch == 1 or epoch % opt.save_epoch_freq == 0))

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            
            # Calculate and save FID score
            if total_iters % opt.fid_freq == 0: # calculate FID score every fid_freq iterations
                with torch.no_grad():
                    model.forward()
                    real_images = model.real_A if opt.direction == 'AtoB' else model.real_B
                    generated_images = model.fake_B if opt.direction == 'AtoB' else model.fake_A
                    fid = calculate_fid(real_images, generated_images, opt.batch_size, opt.device)
                    print('FID score at iteration %d: %f' % (total_iters, fid))
                    fid_scores.append(fid)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        visualizer.close()

def calculate_fid(real_images, generated_images, batch_size, device):
    real_resized = [Resize((299, 299))(img) for img in real_images]
    gen_resized = [Resize((299, 299))(img) for img in generated_images]

    real_resized_tensor = torch.stack(real_resized).to(device)
    gen_resized_tensor = torch.stack(gen_resized).to(device)

    fid = fid_score.calculate_fid_given_paths(
        real_resized_tensor,
        gen_resized_tensor,
        batch_size=batch_size,
        device=device
    )
    return fid

