from Blackjack_counting import Deck, Dealer, Agent, MonteCarlo, counting_temp
import matplotlib.pyplot as plt	

# TODO: import counting to 1000 games

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
money = 10000
wins = 0
loses = 0
draws = 0
x = [0]
y = [money]


# [1] play 1000 games
for i in range(1000):
	# initialize
	if len(deck.card_deck) < 15:
		deck2 = Deck()
		deck.card_deck = deck2.card_deck + deck.card_deck

	dealer.reset()
	agent.reset()
	agent.hit(deck)
	agent.hit(deck)
	dealer.hit(deck)
	dealer.hit(deck)

	showed = dealer.show()
	done = False

	# play 1 game
	while not done:
		# if less than 12, get one more card
		sums = agent.calculate_sum()
		if sums < 12:
			agent.hit(deck)
			continue

		print (counting_temp)
		state = (sums, bool(agent.usable_ace), showed)
		action = agent.policy(state)
		done, reward = dealer.observation(action, agent, deck)

	# update result
	if reward is WIN:
		wins += 1
	elif reward is LOSE:
		loses += 1
	elif reward is DRAW:
		draws += 1

	# update money
	money += 10 * reward

	# update lists
	x.append(i + 1)
	y.append(money)

# [2] show win rate
print("========== Testing : Episode 1000 ===========")
print("TOTAL Games WIN :", wins, "DRAW :", loses, "LOSS :", draws)
print("Total win rate : {:.3f}%".format(wins / (wins + loses) * 100))

# [3] plot money
plt.plot(x, y)
plt.show()


