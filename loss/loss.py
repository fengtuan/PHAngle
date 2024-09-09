import torch

class MSE_Loss(torch.nn.Module):
    def __init__(self,mu=0.5):
        super(MSE_Loss, self).__init__()
        self.mu=mu        
    def forward(self,X, targets):     
        PHI_targets=targets[:,:,0].flatten()
        PSI_targets=targets[:,:,1].flatten()
        PHI_masks=torch.isnan(PHI_targets).logical_not()
        PSI_masks=torch.isnan(PSI_targets).logical_not()  
        PHI_pred=torch.masked_select(X[:,:,0].flatten(),PHI_masks)
        PSI_pred=torch.masked_select(X[:,:,1].flatten(),PSI_masks)
        PHI_targets=torch.masked_select(PHI_targets,PHI_masks)
        PSI_targets=torch.masked_select(PSI_targets,PSI_masks)        
        diff_PHI=torch.abs(PHI_targets-PHI_pred)
        diff_PHI=torch.where(diff_PHI>1.0, 2.0-diff_PHI,diff_PHI)
        diff_PHI=diff_PHI.square()
         
        diff_PSI=torch.abs(PSI_targets-PSI_pred)
        diff_PSI=torch.where(diff_PSI>1.0, 2.0-diff_PSI,diff_PSI)
        diff_PSI=diff_PSI.square()
       
       
        loss=self.mu*diff_PHI.mean()+(1-self.mu)*diff_PSI.mean()         
        return loss


class MAE_Loss(torch.nn.Module):
    def __init__(self,mu=0.5):
        super(MAE_Loss, self).__init__()
        self.mu=mu        
    def forward(self,X, targets):     
        PHI_targets=targets[:,:,0].flatten()
        PSI_targets=targets[:,:,1].flatten()
        PHI_masks=torch.isnan(PHI_targets).logical_not()
        PSI_masks=torch.isnan(PSI_targets).logical_not()  
        PHI_pred=torch.masked_select(X[:,:,0].flatten(),PHI_masks)
        PSI_pred=torch.masked_select(X[:,:,1].flatten(),PSI_masks)
        PHI_targets=torch.masked_select(PHI_targets,PHI_masks)
        PSI_targets=torch.masked_select(PSI_targets,PSI_masks)        
        diff_PHI=torch.abs(PHI_targets-PHI_pred)
        diff_PHI=torch.where(diff_PHI>1.0, 2.0-diff_PHI,diff_PHI)  
         
        diff_PSI=torch.abs(PSI_targets-PSI_pred)
        diff_PSI=torch.where(diff_PSI>1.0, 2.0-diff_PSI,diff_PSI) 
       
       
        loss=self.mu*diff_PHI.mean()+(1-self.mu)*diff_PSI.mean()         
        return loss

class MAE_Loss_SinCos(torch.nn.Module):
    def __init__(self,mu=0.5):
        super(MAE_Loss_SinCos, self).__init__()
        self.mu=mu        
        
    def forward(self,X, targets):     
        PHI_targets=targets[:,:,0].flatten()
        PSI_targets=targets[:,:,1].flatten()
        PHI_masks=torch.isnan(PHI_targets).logical_not()
        PSI_masks=torch.isnan(PSI_targets).logical_not()       
        PHI_pred_sin=torch.masked_select(X[:,:,0].flatten(),PHI_masks)
        PHI_pred_cos=torch.masked_select(X[:,:,1].flatten(),PHI_masks)
        PSI_pred_sin=torch.masked_select(X[:,:,2].flatten(),PSI_masks)
        PSI_pred_cos=torch.masked_select(X[:,:,3].flatten(),PSI_masks)        
        with torch.no_grad():
            PHI_targets=torch.masked_select(PHI_targets,PHI_masks)
            PHI_targets_rad=torch.deg2rad(180*PHI_targets)
            target_PHI_sin=torch.sin(PHI_targets_rad)
            target_PHI_cos=torch.cos(PHI_targets_rad)
            
            PSI_targets=torch.masked_select(PSI_targets,PSI_masks)
            PSI_targets_rad=torch.deg2rad(180*PSI_targets)
            target_PSI_sin=torch.sin(PSI_targets_rad)
            target_PSI_cos=torch.cos(PSI_targets_rad) 
            
        diff_PHI_sin=torch.abs(target_PHI_sin-PHI_pred_sin)
        diff_PHI_cos=torch.abs(target_PHI_cos-PHI_pred_cos) 

        diff_PSI_sin=torch.abs(target_PSI_sin-PSI_pred_sin)
        diff_PSI_cos=torch.abs(target_PSI_cos-PSI_pred_cos)
        
      
        loss=self.mu*(diff_PHI_sin+diff_PHI_cos).mean()+(1-self.mu)*(diff_PSI_sin+diff_PSI_cos).mean()
        return loss
    
class MSE_Loss_SinCos(torch.nn.Module):
    def __init__(self, mu=0.5):
        super(MSE_Loss_SinCos, self).__init__()
        self.mu=mu        
    def forward(self,X, targets):     
        PHI_targets=targets[:,:,0].flatten()
        PSI_targets=targets[:,:,1].flatten()
        PHI_masks=torch.isnan(PHI_targets).logical_not()
        PSI_masks=torch.isnan(PSI_targets).logical_not()        
        PHI_pred_sin=torch.masked_select(X[:,:,0].flatten(),PHI_masks)
        PHI_pred_cos=torch.masked_select(X[:,:,1].flatten(),PHI_masks)
        PSI_pred_sin=torch.masked_select(X[:,:,2].flatten(),PSI_masks)
        PSI_pred_cos=torch.masked_select(X[:,:,3].flatten(),PSI_masks)        
        with torch.no_grad():
            PHI_targets=torch.masked_select(PHI_targets,PHI_masks)
            PHI_targets_rad=torch.deg2rad(180*PHI_targets)
            target_PHI_sin=torch.sin(PHI_targets_rad)
            target_PHI_cos=torch.cos(PHI_targets_rad)
            
            PSI_targets=torch.masked_select(PSI_targets,PSI_masks)
            PSI_targets_rad=torch.deg2rad(180*PSI_targets)
            target_PSI_sin=torch.sin(PSI_targets_rad)
            target_PSI_cos=torch.cos(PSI_targets_rad) 
            
        diff_PHI_sin=torch.square(target_PHI_sin-PHI_pred_sin)
        diff_PHI_cos=torch.square(target_PHI_cos-PHI_pred_cos) 

        diff_PSI_sin=torch.square(target_PSI_sin-PSI_pred_sin)
        diff_PSI_cos=torch.square(target_PSI_cos-PSI_pred_cos)
        
      
        loss=self.mu*(diff_PHI_sin+diff_PHI_cos).mean()+(1-self.mu)*(diff_PSI_sin+diff_PSI_cos).mean()
        return loss