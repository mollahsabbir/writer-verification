import torch
from models import ConvNet
from torchvision.transforms import transforms
class WriterVerifier:
    def __init__(self, model_path):
        model_data = torch.load(model_path)
        
        model = ConvNet(num_channels=model_data['num_channels']
                        , classes=model_data['classes']
                        , embedding_size=model_data['embedding_size']
                        , inference=True)
        
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        self.model = model
        
    def get_embedding(self, image):
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        img_tensor = transform(image)
        print(img_tensor.shape)
        

    def get_score(self, image_1, image_2):
        # TODO: Implement cos_similarity_score
        return 0.5