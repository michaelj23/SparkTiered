from pyspark import SparkContext
from solver_utils import DWULT
from tictactoe import initial_pos, gen_moves, do_move, primitive

sc = SparkContext()

sc.setLogLevel("ERROR")

generation = 0             # Keep track of what generation we are on
generations = [] # RDD's generated going down the tree.
solved_generations = [] # RDD's generated going up the tree.

def next_moves(state):
    """
    Used going down the tree to generate all states.
    Generate the children for the current generation of this parent state from
    the previous generation.
    state = (pos, (generation, [parent positions], DWULT value, remoteness))
    """
    pos = state[0]
    parent_dwult = state[1][2]
    if parent_dwult != DWULT["U"]:
        return []
    children = []
    for move in gen_moves(pos):
        new_pos = do_move(pos, move)
        game_val = primitive(new_pos)
        remoteness = 0 if game_val != DWULT["U"] else None
        children.append((new_pos, (generation, [pos], game_val, remoteness)))
    return children

def aggregate_parents(data_a, data_b):
    """
    Used going down the tree to generate all states.
    For a child position of the current generation, combine into a list
    all parent positions of the previous generation that generated it.
    """
    generation, a_parents, game_val, remoteness = data_a
    b_parents = data_b[1]
    return (generation, a_parents + b_parents, game_val, remoteness)

def flatmap_parents(state):
    """
    Used going up the tree to solve all states.
    For each parent of the given, solved child state, generate an intermediate state
    for the parent with the child's game value translated to what it means for the
    parent (see child_to_parent_game_value). Also update remoteness as a tuple, where
    the first element is set if the child is a win for the parent, the second
    element is set if the child is a loss, and the third is set if the child is
    a tie.
    """
    generation, parents, game_val, remoteness = state[1]
    game_val_parent = child_to_parent_game_value(game_val)
    remote_tup = None
    if game_val_parent == DWULT["W"]:
        remote_tup = (remoteness + 1, float("-inf"), float("-inf"))
    elif game_val_parent == DWULT["L"]:
        remote_tup = (float("inf"), remoteness + 1, float("-inf"))
    elif game_val_parent == DWULT["T"]:
        remote_tup = (float("inf"), float("-inf"), remoteness + 1)
    return [(parent, (generation - 1, [], child_to_parent_game_value(game_val), remote_tup)) \
                for parent in parents]

def child_to_parent_game_value(child_game_val):
    """
    Used going up the tree to solve all states.
    To solve a parent's game value, we need to know what each solved child's
    game value means for it. For example, if a child's game value is a loss, that
    child represents a win for the parent.
    """
    parent_game_val = DWULT["L"]
    if child_game_val == DWULT["L"]:
        parent_game_val = DWULT["W"]
    elif child_game_val == DWULT["T"]:
        parent_game_val = DWULT["T"]
    return parent_game_val

def reduce_by_game_value(data_a, data_b):
    """
    Used going up the tree to solve all states.
    Reduce all of a parent's child game values to solve for the parent's game
    value. Take the min of all remotenesses that are the first tuple element
    (i.e. set by all children that are a win for a parent), the max of
    all remotenesses that are the second tuple element (set by all children
    that are a loss for a parent), and the max of all remotenesses that are the
    third tuple element (set by all children that are a tie for a parent).
    """
    game_val_a = data_a[2]
    game_val_b = data_b[2]
    generation = data_a[0]
    remote_w_a, remote_l_a, remote_t_a = data_a[3]
    remote_w_b, remote_l_b, remote_t_b = data_b[3]
    agg_remote_w = remote_w_a if remote_w_a < remote_w_b else remote_w_b
    agg_remote_l = remote_l_a if remote_l_a > remote_l_b else remote_l_b
    agg_remote_t = remote_t_a if remote_t_a > remote_t_b else remote_t_b
    agg_remote = (agg_remote_w, agg_remote_l, agg_remote_t)
    if game_val_a == DWULT["W"] or game_val_b == DWULT["W"]:
        # If there is at least one winning move for the parent, its game value
        # is a win
        return (generation, [], DWULT["W"], agg_remote)
    if game_val_a == DWULT["L"] and game_val_b == DWULT["L"]:
        # If all child states represent a loss for the parent, its game value
        # is a loss
        return (generation, [], DWULT["L"], agg_remote)
    if game_val_a == DWULT["T"] or game_val_b == DWULT["T"]:
        # If one child state is a tie for the parent and their are no winning moves,
        # the parent is a tie
        return (generation, [], DWULT["T"], agg_remote)

def determine_remoteness(intermed_state):
    """
    Used going up the tree, after solving for game values. If a node's solved
    game value is a win, set the remoteness to the first tuple value, 1 +
    the min remoteness of the children of all winning moves. If it is
    a loss, set remoteness to the second value, 1 + the max remoteness of the
    children of all losing moves. If it is a tie, set remoteness to the third
    value, 1 + the max remoteness of the children of all tieing moves.
    """
    data = intermed_state[1]
    solved_game_val = data[2]
    remote_tup = data[3]
    solved_remote = None
    if solved_game_val == DWULT["W"]:
        solved_remote = remote_tup[0]
    elif solved_game_val == DWULT["L"]:
        solved_remote = remote_tup[1]
    elif solved_game_val == DWULT["T"]:
        solved_remote = remote_tup[2]
    return (intermed_state[0], (data[0], data[1], solved_game_val, solved_remote))

def merge_data(data_a, data_b):
    """
    Used going up the tree to solve all states.
    Merge intermediate states for a tier/generation that have the solved game
    values for the tier's positions with the originally generated unsolved states
    from going down the tree.
    (pos, (gen, [], solved_value, remoteness)) + (pos, (gen, [pos's parents], unsolved, None))
    = (pos, (gen, [pos's parents], solved_value, remoteness))
    """
    game_val = data_a[2] if data_a[2] != DWULT["U"] else data_b[2]
    remoteness = data_a[3] if data_a[3] is not None else data_b[3]
    return (data_a[0], data_a[1] + data_b[1], game_val, remoteness)

init_pos = initial_pos()
init_val = primitive(init_pos)
init_remote = 0 if init_val != DWULT["U"] else None
next_gen = sc.parallelize([(initial_pos(), (0, [], init_val, init_remote))])

### Go down the tree, generating all (potentially unsolved) states ###
while next_gen.count() > 0:
    generations.append(next_gen)
    generation += 1
    next_gen = next_gen.flatMap(next_moves).reduceByKey(aggregate_parents)

### Go up the tree, solving all states ###
intermed_RDD = None
gen_length = len(generations)
num_gen = gen_length
while num_gen > 0:
    gen_RDD = generations[num_gen - 1]
    if intermed_RDD is not None:
        gen_RDD = gen_RDD.union(intermed_RDD).reduceByKey(merge_data)
    gen_RDD.saveAsPickleFile("solver_tier_" + str(num_gen - 1))
    print("solved gen" + str(num_gen - 1));
    solved_generations.append(gen_RDD)
    intermed_RDD = gen_RDD.flatMap(flatmap_parents).reduceByKey(reduce_by_game_value) \
                            .map(determine_remoteness)
    num_gen -= 1

count_positions = 0
for i in range(gen_length):
    tier = sc.pickleFile("solver_tier_" + str(i))
    if i == 0: # only 1 element in tier 0
        tier_0 = tier.collect()
        initial_state = tier_0[0]
        game_val = initial_state[1][2]
        remoteness = initial_state[1][3]
        print(str(game_val) + " in " + str(remoteness))
    print("size of tier", i);
    size = tier.count()
    print(size)
    count_positions += size
print("Total number of positions: " + str(count_positions))
