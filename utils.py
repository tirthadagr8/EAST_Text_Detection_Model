import matplotlib.pyplot as plt

def display(display_list):
    plt.figure(figsize=(5,5))
    title=[]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
#         plt.title(title[i])
        x=display_list[i].permute(1,2,0).detach().cpu().numpy()
        plt.imshow((x),cmap='gray')
        plt.axis('off')
    plt.show()