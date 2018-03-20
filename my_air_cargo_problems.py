from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        def generate_variables():
            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        yield cargo, plane, airport

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            for c, p, a in generate_variables():
                precond_pos = [At(c, a), At(p, a)]
                precond_neg = []
                effect_add = [In(c, p)]
                effect_rem = [At(c, a)]

                load = Action(Load(c, p, a),
                              [precond_pos, precond_neg],
                              [effect_add, effect_rem])
                loads.append(load)
            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            for c, p, a in generate_variables():
                precond_pos = [In(c, p), At(p, a)]
                precond_neg = []
                effect_add = [At(c, a)]
                effect_rem = [In(c, p)]

                unload = Action(Unload(c, p, a),
                                [precond_pos, precond_neg],
                                [effect_add, effect_rem])
                unloads.append(unload)
            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        current_state = decode_state(state, self.state_map)

        possible_actions = []
        for action in self.actions_list:
            if all([pos in current_state.pos for pos in action.precond_pos]) \
                    and all([neg in current_state.neg for neg in action.precond_neg]):
                possible_actions.append(action)

        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        current_state = decode_state(state, self.state_map)
        for effect in action.effect_add:
            current_state.neg.remove(effect)
            current_state.pos.append(effect)

        for effect in action.effect_rem:
            current_state.pos.remove(effect)
            current_state.neg.append(effect)

        return encode_state(current_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        current_state = decode_state(node.state, self.state_map)
        count = set(self.goal).difference(set(current_state.pos))
        count = len(count)
        return count


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


C1, C2, C3, C4 = 'C1', 'C2', 'C3', 'C4'
P1, P2, P3 = 'P1', 'P2', 'P3'
JFK, SFO, ATL, ORD = 'JFK', 'SFO', 'ATL', 'ORD'


def get_negative_cases(pos, cargos, planes, airports):
    all_cases = [At(c, a) for c in cargos for a in airports]
    all_cases += [At(p, a) for p in planes for a in airports]

    in_cases = [In(c, p) for c in cargos for p in planes]

    neg = set(all_cases).difference(set(pos))
    return list(neg) + in_cases


def air_cargo_p2() -> AirCargoProblem:
    cargos = [C1, C2, C3]
    planes = [P1, P2, P3]
    airports = [JFK, SFO, ATL]

    pos = [At(C1, SFO), At(C2, JFK), At(C3, ATL),
           At(P1, SFO), At(P2, JFK), At(P3, ATL)]

    neg = get_negative_cases(pos, cargos, planes, airports)

    init = FluentState(pos, neg)
    goal = [At(C1, JFK), At(C2, SFO), At(C3, SFO)]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    cargos = [C1, C2, C3, C4]
    planes = [P1, P2]
    airports = [JFK, SFO, ATL, ORD]

    pos = [At(C1, SFO), At(C2, JFK), At(C3, ATL), At(C4, ORD),
           At(P1, SFO), At(P2, JFK)]

    neg = get_negative_cases(pos, cargos, planes, airports)

    init = FluentState(pos, neg)
    goal = [At(C1, JFK), At(C3, JFK), At(C2, SFO), At(C4, SFO)]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def At(x, y):
    # return expr(f'At({x}, {y})')
    return expr('At({}, {})'.format(x, y))


def In(x, y):
    # return expr(f'In({x}, {y})')
    return expr('In({}, {})'.format(x, y))


def Load(c, p, a):
    # return expr(f'Load({c}, {p}, {a})')
    return expr('Load({}, {}, {})'.format(c, p, a))


def Unload(c, p, a):
    # return expr(f'Unload({c}, {p}, {a})')
    return expr('Unload({}, {}, {})'.format(c, p, a))