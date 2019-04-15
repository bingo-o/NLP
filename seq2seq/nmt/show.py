import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with color bar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    
    # Set up axes
    ax.set_xticklabels(['']+input_sentence.split(' ')+['<eos>'], rotation=90)
    ax.set_yticklabels(['']+output_words)
    
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.show()