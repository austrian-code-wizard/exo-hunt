import torch
from torch.utils.data import DataLoader

# Faster RCNN only supports batch size of 1
STATIC_BATCH_SIZE = 1


def overlap(pred_box, true_box):
    # xmin, ymin, xmax, ymax
    true_center = ((true_box[0] + true_box[2]) / 2, (true_box[1] + true_box[3]) / 2)
    width_val = true_center[0] > pred_box[0] and true_center[0] < pred_box[2]
    height_val = true_center[1] > pred_box[1] and true_center[1] < pred_box[3]
    return width_val and height_val


def grade_boxes(pred_boxes, true_boxes):
    successes = 0
    for pred in pred_boxes:
        possibilities = 0
        for true in true_boxes:
            if overlap(pred, true):
                possibilities += 1
        if possibilities:
            successes += 1
    return successes


def check_accuracy(dataset, model, collate_fn, device):
    loader = DataLoader(dataset, batch_size=STATIC_BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        counter = 0
        for imgs, targets in loader:
            imgs = list(image.to(device) for image in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(imgs)
            
            print(f'Iter {counter}')
            pred_boxes = predictions[0]["boxes"]
            tg_boxes = targets[0]['boxes']
            
            if grade_boxes(pred_boxes, tg_boxes) > 0:
                num_correct += 1
            num_samples += 1
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def train(model, optimizer, scheduler, dataset, collate_fn, device, epochs=1):
    """
    Train a model PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(f'Beginning epoch {e + 1}')
        loss = None
        model.train()  # put model to training mode
        data_loader = DataLoader(dataset, batch_size=STATIC_BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
        counter = 0
        for imgs, targets in data_loader:
            
            imgs = list(image.to(device) for image in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())
            if not counter % 10: print(f'Iter {counter}: Loss = {loss}')
            counter += 1

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
        scheduler.step()
        print(f"Finished epoch (Loss: {loss})")


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    targets=list()

    for i, t in batch:
        images.append(i)
        targets.append(t)
    images = torch.stack(images, dim=0)

    return images, targets
