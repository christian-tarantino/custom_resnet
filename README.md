```text
8888888b.                   888b    888 8888888888 888    
888   Y88b                  8888b   888 888        888    
888    888                  88888b  888 888        888    
888   d88P .d88b.  .d8888b  888Y88b 888 8888888    888888 
8888888P" d8P  Y8b 88K      888 Y88b888 888        888    
888 T88b  88888888 "Y8888b. 888  Y88888 888        888    
888  T88b Y8b.          X88 888   Y8888 888        Y88b.  
888   T88b "Y8888   88888P' 888    Y888 8888888888  "Y888 

```
### Main Results on CIFAR-10

| Method | Params (M) | Test Acc (%) |
| :---: | :---: | :---: |
| ResNet-18 (He et al.) | 11.17 | 93.02 |
| **Custom ResNet (Ours)** | **SiLU + Custom Prep** | **4.82** |

*Note: All models were trained for 100 epochs using AdamW optimizer with Cosine Annealing learning rate schedule.*                                                         
                                                          
