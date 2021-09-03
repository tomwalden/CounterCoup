# CounterCoup
An agent for playing _Coup_ via Deep Counterfactual Regret Minimisation

## Contents of packages

### model

The model used to play _Coup_ automatically.

**game_info.py** - The parts of a _Coup_ game that are only visible to the player

**game.py** - A complete model for the card game Coup

**action.py** - Base class for actions

**card.py** - Base class for cards

**exceptions.py** - Exceptions raised by the model

**hand.py** - Representation of a hand in Coup. Not used in the implementation per se - easier to use lists - but useful when doing manipulation.

**history.py** - provides a history of every move played during the game

**player.py** - Class for each Coup player

**state.py** - Base class for Coup game states

#### model/items

Items (e.g. actions, cards) used by the Model

**actions.py** - action classes that plays can play in _Coup_

**cards.py** - card classes representing cards in _Coup_

**states.py** - game states in the model

### trainer

The module used to train the _CounterCoup_ strategy networks

**trainer.py** - Overarching class for generating the neural networks needed for Deep CFR in CounterCoup

**trainer_stats.py** - Holder for stats we pick up whilst training

**traverser.py** - Base class for traversers

#### trainer/traversers

Traverser modules used to implement specific sampling methods

**full_robust.py** - Full robust traversals, with no narrowing down/drawing

**limited_robust.py** - Limited robust traversals

**outcome.py** - Traverser for outcome sampling - select one action per turn

**strategy_optimised.py** - Traverser for strategy optimised robust sampling

**timid_biased.py** - Traverser that defaults to a timid profile for zero regrets

### player

The module used to test the strategy networks

**agent.py** - Base class for agents

**measure_tools.py** - Tools used to measure the effectiveness of the agents

**online_play.py** - Play CounterCoup using ChickenKoup

**self_play.py** - Have several agents play Coup against each other

#### player/agents

The agents that we can use to play _Coup_

**random.py** - Completely random agent

**countercoup.py** - Agent that plays according to our CounterCoup networks

**timid.py** - Agent that never bluffs, and never blocks, but always counteracts if it has the card

**aggressive.py** - An agent that is more aggressive than a Random agent

### shared

Objects and classes that are shared between modules

**batch_memory.py** - Sequence based object that can batch up the data for each epoch when training

**infoset.py** - Represents the information set at a given stage of the game, in a compacted form that can be fed into a neural network

**memory.py** - Memory for the trainer, utilising reservoir sampling

**net_group.py** - Combined group of networks used to define CounterCoup

**network.py** - Base class for the neural networks used in Deep CFR

**structure.py** - Base class for network structures

**tools.py** - Tools shared by everything in CounterCoup

#### shared/networks

Specific network types that subclass Network. Differ on the output layer

**action_net.py** - Network used for action decisions

**block_counteract_net.py** - Network used for counteraction and blocking decisions

**lose_net.py** - Network used for discards and loses

### shared/structures

Underlying neural network structures that we tested

**basic.py** - Basic structure, 6 layers with no recurrent cells

**lstm.py** - Default structure, 6 layers with LSTM cells

**relu.py** - Default structure but with ReLU activation

**enhanced.py** - Enhanced structure, 10 layers plus sigmoid layers after LSTM cells

**enhanced_basic.py** - Basic enhanced structure, same as Enhanced structure but without LSTM cells