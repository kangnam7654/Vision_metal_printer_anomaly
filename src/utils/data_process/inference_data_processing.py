def image_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    image = torch.permute(image, (2, 0, 1))
    image = image.unsqueeze(0)
    image = image / 255
    image = image.float().cuda()
    return image
