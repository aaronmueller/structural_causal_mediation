from collections import defaultdict

def read_grammar(grammar_file):
    grammar = {}
    with open(grammar_file, 'r') as open_grammar:
        for line in open_grammar:
            if len(line.strip()) < 4:
                continue
            nt = Nonterminal(name="", att=set())
            nt.read(line)
            grammar[(nt.name, frozenset(nt.attributes))] = nt.terminals

    return grammar


class Nonterminal:
    def __init__(self, name="", att=set(), terminals=[]):
        self.name = name
        self.attributes = att
        self.terminals = terminals

    # define __eq__ and __hash__ s.t. Nonterminals with the same
    # name and attributes are treated as equivalent
    def __eq__(self, other):
        if not isinstance(other, Nonterminal):
            return NotImplemented
        return self.name == other.name and self.attributes == other.attributes
                                            
    def __hash__(self):
        return hash((self.name, frozenset(self.attributes)))

    # input: a grammar line
    def read(self, in_str):
        nt_name = in_str.split("[")[0].strip()
        # read between brackets
        nt_attributes = in_str[in_str.find("[")+1:in_str.find("]")]
        # read terminals corresponding to nonterminal
        nt_terminals = []
        nt_terminal_list = in_str.split("]")[1].split("|")
        for nt_terminal in nt_terminal_list:
            nt_terminals.append(nt_terminal.strip())

        self.name = nt_name
        self.terminals = nt_terminals
        if not nt_attributes:
            return
        elif "," not in nt_attributes:
            self.attributes.update([nt_attributes.strip()])
            return
        for attribute in nt_attributes.split(","):
            self.attributes.update([attribute.strip()])
