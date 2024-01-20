import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models
from .load_config import load_config, save_config
from config import BASE_DIR, CONFIG_DIR, LORA_CONFIG, LORA
from peft import get_peft_model

cudnn.benchmark = True



def load_model():
    # load configuration
    config = load_config(CONFIG_DIR)
    MODEL_NAME = config['MODEL']['MODEL_NAME'] if config['MODEL']['MODEL_NAME'] else 'resnet50'
    CHECKPOINT = config['MODEL']['CHECKPOINT'] if config['MODEL']['CHECKPOINT'] else ''
    if CHECKPOINT:
        CHECKPOINT = os.path.join(BASE_DIR, CHECKPOINT)
    NUMCLASS = config['MODEL']['NUMCLASS'] if config['MODEL']['NUMCLASS'] else 2


    try:
        if MODEL_NAME == "efficientnetb7":
            if CHECKPOINT:
                model = models.efficientnet_b7(weights=False)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.efficientnet_b7(weights=True)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

            for param in model.parameters():
                param.requires_grad = False

            # mo 2 lop cuoi cua feature
            model.features[-2:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "efficientnetb6":
            if CHECKPOINT:
                model = models.efficientnet_b6(weights=False)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.efficientnet_b6(weights=True)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

            for param in model.parameters():
                param.requires_grad = False

            # mo 2 lop cuoi cua feature
            model.features[-2:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "efficientnetb5":
            if CHECKPOINT:
                model = models.efficientnet_b5(weights=False)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.efficientnet_b5(weights=True)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

            for param in model.parameters():
                param.requires_grad = False

            # mo 2 lop cuoi cua feature
            model.features[-2:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "efficientnetb4":
            if CHECKPOINT:
                model = models.efficientnet_b4(weights=False)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.efficientnet_b4(weights=True)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

            for param in model.parameters():
                param.requires_grad = False

            # mo 2 lop cuoi cua feature
            model.features[-2:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "efficientnetb3":
            if CHECKPOINT:
                model = models.efficientnet_b3(weights=False)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.efficientnet_b3(weights=True)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

            for param in model.parameters():
                param.requires_grad = False

            # mo 2 lop cuoi cua feature
            model.features[-2:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "efficientnetb2":
            if CHECKPOINT:
                model = models.efficientnet_b2(weights=False)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.efficientnet_b2(weights=True)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

            for param in model.parameters():
                param.requires_grad = False

            # mo 2 lop cuoi cua feature
            model.features[-2:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "efficientnetb1":
            if CHECKPOINT:
                model = models.efficientnet_b1(weights=False)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.efficientnet_b1(weights=True)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

            for param in model.parameters():
                param.requires_grad = False

            # mo 2 lop cuoi cua feature
            model.features[-2:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "efficientnetb0":
            if CHECKPOINT:
                model = models.efficientnet_b0(weights=False)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.efficientnet_b0(weights=True)
                model.classifier = nn.Linear(model.classifier[1].in_features, NUMCLASS)

            for param in model.parameters():
                param.requires_grad = False

            # mo 2 lop cuoi cua feature
            model.features[-2:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        


        # ====================================================
        # ======= RESNET =====================================

        elif MODEL_NAME == "resnet18":            
            if CHECKPOINT:
                model = models.resnet18(weights=True)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)
                    
                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                if LORA:
                    model = get_peft_model(model, LORA_CONFIG)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.resnet18(weights=True)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)
                if LORA:
                    model = get_peft_model(model, LORA_CONFIG)

            for param in model.parameters():
                    param.requires_grad = False

            # mo lop cuoi cua feature
            model.layer4.requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.fc.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "resnet34":            
            if CHECKPOINT:
                model = models.resnet34(weights=False)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.resnet34(weights=True)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo lop cuoi cua feature
            model.layer4.requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.fc.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "resnet50":            
            if CHECKPOINT:
                model = models.resnet50(weights=False)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.resnet50(weights=True)
                # print(model)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo lop cuoi cua feature
            model.layer4.requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.fc.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "resnet101":            
            if CHECKPOINT:
                model = models.resnet101(weights=False)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.resnet101(weights=True)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo lop cuoi cua feature
            model.layer4.requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.fc.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "resnet152":            
            if CHECKPOINT:
                model = models.resnet152(weights=False)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.resnet152(weights=True)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo lop cuoi cua feature
            model.layer4.requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.fc.parameters():
                param.requires_grad = True
        
        # =====================================================
        # ============== VGG ==================================

        elif MODEL_NAME == "vgg19":
            if CHECKPOINT:
                model = models.vgg19(weights=False)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.vgg19(weights=True)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo 2 lop cuoi cua feature
            model.features[30:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "vgg19bn":
            if CHECKPOINT:
                model = models.vgg19_bn(weights=False)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.vgg19_bn(weights=True)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo 2 lop cuoi cua feature
            model.features[46:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "vgg16":
            if CHECKPOINT:
                model = models.vgg16(weights=False)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.vgg16(weights=True)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo 2 lop cuoi cua feature
            model.features[24:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "vgg16bn":
            if CHECKPOINT:
                model = models.vgg16_bn(weights=False)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.vgg16_bn(weights=True)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo 2 lop cuoi cua feature
            model.features[37:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "vgg13":
            if CHECKPOINT:
                model = models.vgg13(weights=False)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.vgg13(weights=True)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo 2 lop cuoi cua feature
            model.features[20:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "vgg13bn":
            if CHECKPOINT:
                model = models.vgg13_bn(weights=False)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.vgg13_bn(weights=True)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo 2 lop cuoi cua feature
            model.features[28:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "vgg11":
            if CHECKPOINT:
                model = models.vgg11(weights=False)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.vgg11(weights=True)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo 2 lop cuoi cua feature
            model.features[16:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "vgg11bn":
            if CHECKPOINT:
                model = models.vgg11_bn(weights=False)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.vgg11_bn(weights=True)
                model.classifier[6] = nn.Linear(model.classifier[-1].in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo 2 lop cuoi cua feature
            model.features[22:].requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        # ====================================================
        # ============= GOOGLENET ============================

        elif MODEL_NAME == "googlenet":
            if CHECKPOINT:
                model = models.googlenet(weights=False)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.googlenet(weights=True)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

            for param in model.parameters():
                    param.requires_grad = False
                    
            # mo 2 lop cuoi cua feature
            model.inception5a.requires_grad_(True)
            model.inception5b.requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.fc.parameters():
                param.requires_grad = True


        #  update config
        config['MODEL']['MODEL_NAME'] = MODEL_NAME
        config['MODEL']['CHECKPOINT'] = CHECKPOINT
        config['MODEL']['NUMCLASS'] = NUMCLASS
        save_config(config, 'config.yaml')

        return model
    except:
        print('Error: Could not load model.')
        exit(1)