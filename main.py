from Blackjack import Deck, Dealer, Agent, MonteCarlo
import matplotlib.pyplot as plt

# Const
WIN = 1
DRAW = 0
LOSE = -1

# class initialize
dealer = Dealer()
agent = Agent()
deck = Deck()

# train 1M iters
mc = MonteCarlo()
mc.train(dealer, agent, deck, 1000000, False)

# initialize variable
current = 10000
wins = 0
x = [0]
y = [current]

# play 1000 games
for i in range(1000):
	returned = mc.generate_episode(dealer, agent, deck)
	result = returned[-1][-1]

	if result is WIN:
		wins += 1
	current += (10 * result)
	
	x.append(i)
	y.append(current)

# show win rate, money plot
print ("WIN RATE: {:.3f}%".format(wins / 10))
plt.plot(x, y)
plt.show()
