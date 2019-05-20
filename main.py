from Blackjack import Deck, Dealer, Agent, MonteCarlo


# Homework1
dealer = Dealer()
agent = Agent()
deck = Deck()

mc = MonteCarlo()
mc.train(dealer, agent, deck, 1000000)

print (len(agent.Q_table))
for k, v in agent.Q_table.items():
	print (k, ':', v)
