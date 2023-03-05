import torch
import models
import importlib
importlib.reload(models)
from torchvision.transforms import transforms
from torch import nn

class WriterVerifier:
    def __init__(self, model_path):
        model_data = torch.load(model_path)
        
        model = models.ConvNet(num_channels=model_data['num_channels']
                        , classes=model_data['classes']
                        , embedding_size=model_data['embedding_size']
                        , inference=True)
        
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        self.model = model
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((100,60)),
        ])
        
    def get_embedding(self, image):
        img_tensor = self.transform(image)
        print(img_tensor.shape)
        

    def get_score(self, image_1, image_2):
        img1_t = self.transform(image_1)
        img2_t = self.transform(image_2)
        embedding_1 = self.model(img1_t.unsqueeze(0))
        embedding_2 = self.model(img2_t.unsqueeze(0))
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity_score = cos(embedding_1, embedding_2)

        return similarity_score