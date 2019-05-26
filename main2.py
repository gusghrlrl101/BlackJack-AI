from Blackjack2 import Deck, Dealer, Agent, MonteCarlo
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
mc.train(dealer, agent, deck, 1000000, True)

# initialize variable
current = 10000
wins = 0
loses = 0
x = [0]
y = [current]

# play 1000 games
for i in range(1000):
	returned = mc.generate_episode(dealer, agent, deck, False)
	result = returned[-1][-1]

	if result is WIN:
		wins += 1
	elif result is LOSE:
		loses += 1
	current += (10 * result)
	
	x.append(i)
	y.append(current)

# show win rate, money plot
print ("WIN RATE: {:.3f}%".format(100 * wins / (wins + loses)))
plt.plot(x, y)
plt.show()