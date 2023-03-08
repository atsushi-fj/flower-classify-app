from torchvision import transforms

from model import CCT


def create_cct_model(embedding_dim=256,
                     n_classes=102,
                     img_size=224):
    
    transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    model = CCT(
        embedding_dim=embedding_dim,
        n_classes=n_classes
    )
    
    return model, transforms
