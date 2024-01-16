import torch
from decision_transformer_atari import GPTConfig, GPT

def run_model(vocab_size=18, block_size=90, model_type="reward_conditioned", timesteps=2719):
    # initialize a baby GPT model
    mconf = GPTConfig(vocab_size,  # Vocab size
                      block_size,  # Block size
                      n_layer=6,  # Number of transformer layers
                      n_head=8,  # Number of attention heads
                      n_embd=128,  # Embedding dimension
                      model_type=model_type,  # Reward conditioned or not
                      max_timestep=timesteps)  # Max timesteps in an episode
    model = GPT(mconf)

    model.load_pretrained("checkpoints/Seaquest_123.pth", cpu=True)  # initialize weights from pretrained model
    
    print("Model loaded:")
    print(model)

    ############### code underneath generated with copilot ###############
    # toks are not useful but skeleton of code is  
    
    # initialize a random stream of tokens for the demo
    # toks = torch.randint(0, 32, (1, 128)).long()

    # # run the model forward to predict logits (output scores before softmax)
    # logits = model(toks)  # shape [1, 128, 32]

    # # train loop example saving checkpoints every epoch
    # optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    # for epoch in range(3):
    #     for i in range(100):
    #         logits = model(toks)  # shape [1, 128, 32]
    #         loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), toks.view(-1))
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #     torch.save(model.state_dict(), f'seaquest_model_{epoch}.pth')
    
    
if __name__ == "__main__":
    run_model(vocab_size=18, block_size=90, model_type="reward_conditioned", timesteps=2719)