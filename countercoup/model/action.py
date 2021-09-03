class Action:
    """
    Base class for actions
    """

    # The card associated with this action
    action_card = None

    # Any cards that can counteract the action
    c_action_cards = None

    # Does this action attack another player?
    attack_action = False

    # The coins to be paid to use this action
    cost = 0
