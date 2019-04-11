"""
Based on codegen, see license below

-----
Copyright (c) 2008, Armin Ronacher
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from ast import NodeVisitor, NodeTransformer
from ast import And, Or     
from ast import Eq, Gt, GtE, In, Is, IsNot, Lt, LtE, NotEq, NotIn
from ast import Invert, Not, UAdd, USub
from ast import If, Name
from ast import Add,     Sub,     Mult,     Div,     FloorDiv,     Mod,     LShift,     RShift,     BitOr,     BitAnd,     BitXor,     Pow
    
from collections import defaultdict

EDGE_TYPE = {
    'child': 0,
    'NextToken': 1,
    'last_lexical': 2,
    'last_use': 3,
    'last_write': 4,
    'computed_from': 5,
    'return_to': 6,
}

NODE_TYPE = {
    'non_terminal': 0,
    'terminal': 1,
    'identifier': 2
}

BOOLOP_SYMBOLS = {
    And:        'and',
    Or:         'or'
}

BINOP_SYMBOLS = {
    Add:        '+',
    Sub:        '-',
    Mult:       '*',
    Div:        '/',
    FloorDiv:   '//',
    Mod:        '%',
    LShift:     '<<',
    RShift:     '>>',
    BitOr:      '|',
    BitAnd:     '&',
    BitXor:     '^',
    Pow:        '**'
}

CMPOP_SYMBOLS = {
    Eq:         '==',
    Gt:         '>',
    GtE:        '>=',
    In:         'in',
    Is:         'is',
    IsNot:      'is not',
    Lt:         '<',
    LtE:        '<=',
    NotEq:      '!=',
    NotIn:      'not in'
}

UNARYOP_SYMBOLS = {
    Invert:     '~',
    Not:        'not',
    UAdd:       '+',
    USub:       '-'
}

ALL_SYMBOLS = {}
ALL_SYMBOLS.update(BOOLOP_SYMBOLS)
ALL_SYMBOLS.update(BINOP_SYMBOLS)
ALL_SYMBOLS.update(CMPOP_SYMBOLS)
ALL_SYMBOLS.update(UNARYOP_SYMBOLS)

class AstGraphGenerator(NodeVisitor):
    """
    """

    def __init__(self, use_ast=True):
        self.use_ast = use_ast
        self.identifier_only = False
        self.syntactic_only = False

        self.node_id = 0
        self.graph = defaultdict(set)
        self.node_label = {}
        self.node_type = {}
        self.representations = []

        self.terminal_path = []

        self.parent = None                #For child edges
        self.previous_token = None        #For NextToken edges
        self.last_lexical = {}            #For last_lexical edges

        self.current_function = None      #For return_to edges
        self.is_return = False

        self.assign_context = None

        self.is_revisit = False
        self.lvalue = False
        self.last_access = defaultdict(lambda: [set(), set()])

    # --- ID and Edge manipulation ----

    def __add_edge(self, nid, label=None, edge_type='child'):
        if edge_type == 'child' and self.parent is not None \
                                and self.use_ast and not self.is_revisit:
            self.graph[(self.parent, nid)].add('child')
        if edge_type == 'NextToken' and self.previous_token is not None \
                                     and not self.is_revisit:
            self.graph[(self.previous_token, nid)].add('NextToken')
        if edge_type == 'last_lexical' and label in self.last_lexical \
                                       and not self.is_revisit:
            self.graph[(nid, self.last_lexical[label])].add('last_lexical')
        if edge_type == 'last_use' and not self.syntactic_only:
            for use in self.last_access[label][0]:
                self.graph[(nid, use)].add('last_use')
        if edge_type == 'last_write' and not self.syntactic_only:
            for use in self.last_access[label][1]:
                self.graph[(nid, use)].add('last_write')
        if edge_type == 'return_to' and not self.syntactic_only \
                                        and self.is_return and self.use_ast\
                                        and self.current_function is not None:
            self.graph[(nid, self.current_function)].add('return_to')
        if edge_type == 'computed_from' and self.lvalue\
                                        and self.assign_context is not None:
            for other in self.assign_context:
                self.graph[(nid,other)].add('computed_from')

       
    def __create_node(self, label, node_type):
        self.node_label[self.node_id] = label
        self.node_type[self.node_id] = node_type

        if (node_type == NODE_TYPE['terminal'] or node_type == NODE_TYPE['identifier']) \
           and self.node_id not in self.terminal_path:
            self.terminal_path.append(self.node_id)

        self.node_id += 1
        return self.node_id - 1


    def revisit(self, node, root):
        old_id, old_parent, old_last_lexical, old_previous =  \
                        (self.node_id, self.parent, self.last_lexical, self.previous_token)
        self.node_id = root
        self.is_revisit = True
        super().visit(node)
        end_id = self.node_id
        self.is_revisit = False
        self.node_id, self.parent, self.last_lexical, self. previous_token = \
                        (old_id, old_parent, old_last_lexical,  old_previous)
        return end_id

    # --- Branch and Context Manipulation

    def __deep_copy(self, def_dict):
        return defaultdict(def_dict.default_factory,
                           {key: [read.copy(), write.copy()] for key, (read, write) in def_dict.items()})

    def __add_context(self, new_context):
        return defaultdict(self.last_access.default_factory,
                {label: [set.union(new_context[label][0], self.last_access[label][0]), 
                         set.union(new_context[label][1], self.last_access[label][1])] for label in self.last_access.keys()})


    def __enter_branching(self):
        return self.__deep_copy(self.last_access), self.__deep_copy(self.last_access)
    
    def __new_branch(self, root_context, branched_context):
        branched_context = self.__add_context(branched_context)
        self.last_access = self.__deep_copy(root_context)
        return branched_context
    
    def __leave_branching(self, branched_context):
        self.last_access = self.__add_context(branched_context)
    

    # --- Functions for general types of nodes -----

    def terminal(self, label):
        if not self.identifier_only:
            nid = self.__create_node(label, NODE_TYPE['terminal'])
            
            self.__add_edge(nid, edge_type='child')
            self.__add_edge(nid, edge_type='NextToken')
            self.__add_edge(nid, edge_type='return_to')

            if not self.is_revisit:
                self.previous_token = nid
        else:
            pass

    def non_terminal(self, node):
        if self.use_ast:
            nid = self.__create_node(node.__class__.__name__, NODE_TYPE['non_terminal'])
            self.__add_edge(nid)
            self.parent = nid
        else:
            pass

    def identifier(self, label):
        nid = self.__create_node(label, NODE_TYPE['identifier'])

        self.__add_edge(nid, edge_type='child')
        self.__add_edge(nid, edge_type='NextToken')
        self.__add_edge(nid, label=label, edge_type='last_lexical')

        self.__add_edge(nid, label=label, edge_type='last_use')
        self.__add_edge(nid, label=label, edge_type='last_write')
        self.__add_edge(nid, label=label, edge_type='computed_from')

        if not self.is_revisit:
            self.previous_token = nid
            self.last_lexical[label] = nid
        if not self.syntactic_only:
            self.last_access[label][0] = {nid}
            if self.lvalue:
                self.last_access[label][1] = {nid}

        if self.assign_context is not None and not self.lvalue:
            self.assign_context.add(nid)

    # --- Visitors ---

    def body(self, statements, root=None):
        self.new_line = True
        for stmt in statements:
            if root is None:
                self.visit(stmt)
            else:
                root = self.revisit(stmt, root)
                

    def body_or_else(self, node, root=None):
        self.body(node.body, root=root)
        if node.orelse:
            self.terminal('else')
            self.terminal(':')
            self.body(node.orelse, root=self.node_id if root is not None else None)

    def list_nodes(self, lnodes):
        for idx, nodes in enumerate(lnodes):
            self.terminal(', ' if idx else '')
            self.visit(nodes)

    def signature(self, node):
        want_comma = []
        def write_comma():
            if want_comma:
                self.terminal(', ')
            else:
                want_comma.append(True)

        padding = [None] * (len(node.args) - len(node.defaults))
        for arg, default in zip(node.args, padding + node.defaults):
            write_comma()
            self.visit(arg)
            if default is not None:
                self.terminal('=')
                self.visit(default)
        if node.vararg is not None:
            write_comma()
            self.terminal('*')
            self.terminal(node.vararg.arg)

    def decorators(self, node):
        for decorator in node.decorator_list:
            self.terminal('@')
            self.visit(decorator)

    # - Assign nodes need special attention since for data flow purposes they should be evaluated left to right -

    def visit_Assign(self, node):
        gparent = self.parent
        self.non_terminal(node)
        lside_id = self.node_id
        self.syntactic_only = True
        for idx, target in enumerate(node.targets):
            if idx:
                self.terminal('=')
            self.visit(target)
        self.syntactic_only = False

        self.terminal('=')
        self.assign_context = set()
        self.visit(node.value)

        self.lvalue = True
        for target in node.targets:
            lside_id = self.revisit(target, root=lside_id) + (1 if not self.identifier_only else 0)
        self.lvalue = False
        self.assign_context = None
        self.parent = gparent

    def visit_AugAssign(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        lside_id = self.node_id
        self.syntactic_only = True
        self.visit(node.target)
        self.syntactic_only = False

        self.terminal(BINOP_SYMBOLS[type(node.op)] + '=')
        self.assign_context = set()

        self.visit(node.value)

        self.lvalue = True
        self.revisit(node.target, root=lside_id)
        self.lvalue = False
        self.assign_context = None
        self.parent = gparent

    def visit_ImportFrom(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('from')
        self.terminal('%s%s' % ('.' * node.level, node.module))
        self.terminal('import')
        for idx, item in enumerate(node.names):
            if idx:
                self.terminal(', ')
            self.visit(item)
        self.parent = gparent

    def visit_Import(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        for item in node.names:
            self.terminal('import ')
            self.visit(item)
        self.parent = gparent

    def visit_Expr(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.generic_visit(node)
        self.parent = gparent

    def visit_FunctionDef(self, node):
        gparent = self.parent
        top_function = self.current_function
        self.current_function = self.node_id
        self.non_terminal(node)

        self.decorators(node)
        self.terminal('def')
        self.terminal(node.name)
        self.terminal('(')
        self.signature(node.args)
        self.terminal(')')
        self.terminal(':')


        self.body(node.body)
        self.current_function = top_function
        
        self.parent = gparent

    def visit_ClassDef(self, node):
        gparent = self.parent
        self.non_terminal(node)
        have_args = []
        def paren_or_comma():
            if have_args:
                self.terminal(',')
            else:
                have_args.append(True)
                self.terminal('(')

        self.decorators(node)
        
        self.terminal('class %s' % node.name)
        for base in node.bases:
            paren_or_comma()
            self.visit(base)
        # XXX: the if here is used to keep this module compatible
        #      with python 2.6.
        if hasattr(node, 'keywords'):
            for keyword in node.keywords:
                paren_or_comma()
                self.terminal(keyword.arg + '=')
                self.visit(keyword.value)
        self.terminal(have_args and '):' or ':')
        self.body(node.body)
        self.parent = gparent

    def visit_If(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('if ')
        self.visit(node.test)
        self.terminal(':')

        root_context, branched_context = self.__enter_branching()

        self.body(node.body)

        while node.orelse:
            branched_context = self.__new_branch(root_context, branched_context)
            else_ = node.orelse
            if len(else_) == 1 and isinstance(else_[0], If):
                node = else_[0]
                self.terminal('elif ')
                self.visit(node.test)
                self.terminal(':')
                self.body(node.body)

            else:
                self.terminal('else:')
                self.body(else_)
                break

        #TODO: case of if-else where identifier is used in both branches

        self.__leave_branching(branched_context)
        self.parent = gparent

    def visit_For(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('for ')

        self.lvalue = True
        target_id = self.node_id
        self.visit(node.target)
        self.lvalue = False
        self.terminal('in')
        iter_id = self.node_id
        self.visit(node.iter)
        self.terminal(':')

        body_id = self.node_id
        _, top_branching = self.__enter_branching()
        self.body_or_else(node)
        self.lvalue = True
        self.revisit(node.target, root=target_id)
        self.lvalue = False
        self.revisit(node.iter, root=iter_id)

        _, bottom_branching = self.__enter_branching()
        self.body_or_else(node, root=body_id)
        self.revisit(node.target, root=target_id)
        self.revisit(node.iter, root=iter_id)

        self.__leave_branching(bottom_branching)
        self.__leave_branching(top_branching)

        self.parent = gparent

    def visit_While(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('while ')
        
        test_id = self.node_id
        self.visit(node.test)
        self.terminal(':')

        body_id = self.node_id
        _, top_branching = self.__enter_branching()
        self.body_or_else(node)
        self.revisit(node.test, test_id)


        _, bottom_branching = self.__enter_branching()
        self.body_or_else(node, root=body_id)
        self.revisit(node.test, test_id)

        self.__leave_branching(bottom_branching)
        self.__leave_branching(top_branching)
        
        self.parent = gparent

    def visit_With(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('with')
        self.list_nodes(node.items)
            
        self.terminal(':')
        self.body(node.body)
        self.parent = gparent

    def visit_Pass(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('pass')
        self.parent = gparent

    def visit_Print(self, node):
        gparent = self.parent
        self.non_terminal(node)
        # XXX: python 2.6 only
        
        self.terminal('print')
        want_comma = False
        if node.dest is not None:
            self.terminal('>>')
            self.visit(node.dest)
            want_comma = True
        for value in node.values:
            if want_comma:
                self.terminal(',')
            self.visit(value)
            want_comma = True
        if not node.nl:
            self.terminal(',')
        self.parent = gparent

    def visit_Delete(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('del')
        for idx, target in enumerate(node.targets):
            if idx:
                self.terminal(', ')
            self.visit(target)
        self.parent = gparent

    def visit_TryExcept(self, node):
        gparent = self.parent
        self.non_terminal(node)

        self.terminal('try')
        self.terminal(':')
        self.body(node.body)


        for handler in node.handlers:
            self.visit(handler)
        self.parent = gparent

    def visit_TryFinally(self, node):
        gparent = self.parent
        self.non_terminal(node)

        self.terminal('try')
        self.terminal(':')
        self.body(node.body)
        
        self.terminal('finally')
        self.terminal(':')
        self.body(node.finalbody)
        self.parent = gparent

    def visit_Global(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('global ' + ', '.join(node.names))
        self.parent = gparent

    def visit_Nonlocal(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('nonlocal ' + ', '.join(node.names))
        self.parent = gparent

    def visit_Return(self, node):
        gparent = self.parent

        self.non_terminal(node)
        
        if node.value:
            self.is_return = True
            self.terminal('return')
            self.is_return = False
            self.visit(node.value)

        else:
            self.is_return = True
            self.terminal('return')
            self.is_return = False
        self.parent = gparent

    def visit_Break(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('break')
        self.parent = gparent

    def visit_Continue(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('continue')
        self.parent = gparent

    def visit_Raise(self, node):
        gparent = self.parent
        self.non_terminal(node)
        # XXX: Python 2.6 / 3.0 compatibility
        
        self.terminal('raise')
        if hasattr(node, 'exc') and node.exc is not None:
            self.visit(node.exc)
            if node.cause is not None:
                self.terminal('from')
                self.visit(node.cause)
        elif hasattr(node, 'type') and node.type is not None:
            self.visit(node.type)
            if node.inst is not None:
                self.terminal(',')
                self.visit(node.inst)
            if node.tback is not None:
                self.terminal(',')
                self.visit(node.tback)

        self.parent = gparent

    def visit_Attribute(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.visit(node.value)
        self.terminal('.')
        self.terminal(node.attr)
        self.parent = gparent

    def visit_Call(self, node):
        gparent = self.parent
        self.non_terminal(node)
        want_comma = []
        def write_comma():
            if want_comma:
                self.terminal(', ')
            else:
                want_comma.append(True)

        self.visit(node.func)
        self.terminal('(')
        for arg in node.args:
            write_comma()
            self.visit(arg)
        for keyword in node.keywords:
            write_comma()
            arg = keyword.arg or ''
            self.terminal(arg + '=' if arg else '**')
            self.visit(keyword.value)
        self.terminal(')')
        self.parent = gparent

    def visit_Name(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.identifier(node.id)
        self.parent = gparent

    def visit_NameConstant(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal(str(node.value))
        self.parent = gparent

    def visit_Str(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal("string")
        self.parent = gparent

    def visit_Bytes(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal(repr(node.s))
        self.parent = gparent

    def visit_Num(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal(repr(node.n))
        self.parent = gparent

    def visit_Tuple(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal('(')
        idx = -1
        for idx, item in enumerate(node.elts):
            if idx:
                self.terminal(',')
            self.visit(item)
        self.terminal(idx and ')' or ',)')

    def sequence_visit(left, right):
        def visit(self, node):
            nid = self.non_terminal(node)
            gparent, self.parent = self.parent, nid
            self.terminal(left)
            for idx, item in enumerate(node.elts):
                if idx:
                    self.terminal(', ')
                self.visit(item)
            self.terminal(right)
            self.parent = gparent
        return visit

    visit_List = sequence_visit('[', ']')
    visit_Set = sequence_visit('{', '}')
    del sequence_visit

    def visit_Dict(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal('{')
        for idx, (key, value) in enumerate(zip(node.keys, node.values)):
            if idx:
                self.terminal(',')
            self.visit(key)
            self.terminal(':')
            self.visit(value)
        self.terminal('}')
        self.parent = gparent

    def visit_BinOp(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.visit(node.left)
        self.terminal(' %s ' % BINOP_SYMBOLS[type(node.op)])
        self.visit(node.right)
        self.parent = gparent

    def visit_BoolOp(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal('(')
        for idx, value in enumerate(node.values):
            if idx:
                self.terminal(' %s ' % BOOLOP_SYMBOLS[type(node.op)])
            self.visit(value)
        self.terminal(')')
        self.parent = gparent

    def visit_Compare(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal('(')
        self.visit(node.left)
        for op, right in zip(node.ops, node.comparators):
            self.terminal(' %s ' % CMPOP_SYMBOLS[type(op)])
            self.visit(right)
        self.terminal(')')
        self.parent = gparent

    def visit_UnaryOp(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal('(')
        op = UNARYOP_SYMBOLS[type(node.op)]
        self.terminal(op)
        self.visit(node.operand)
        self.terminal(')')
        self.parent = gparent

    def visit_Subscript(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.visit(node.value)
        self.terminal('[')
        self.visit(node.slice)
        self.terminal(']')
        self.parent = gparent

    def visit_Slice(self, node):
        gparent = self.parent
        self.non_terminal(node)
        if node.lower is not None:
            self.visit(node.lower)
        self.terminal(':')
        if node.upper is not None:
            self.visit(node.upper)
        if node.step is not None:
            self.terminal(':')
            if not (isinstance(node.step, Name) and node.step.id == 'None'):
                self.visit(node.step)
        self.parent = gparent

    def visit_ExtSlice(self, node):
        gparent = self.parent
        self.non_terminal(node)
        for i in range(0, len(node.dims)):
            if i > 0:
                self.terminal(', ')
            self.visit(node.dims[i])
        self.parent = gparent

    def visit_Yield(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal('yield')
        self.visit(node.value)
        self.parent = gparent

    def visit_Lambda(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal('lambda')
        self.signature(node.args)
        self.terminal(':')
        self.visit(node.body)
        self.parent = gparent

    def visit_Ellipsis(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal('Ellipsis')

    def generator_visit(left, right):
        def visit(self, node):
            nid = self.non_terminal(node)
            gparent, self.parent = self.parent, nid
            self.terminal(left)
            self.visit(node.elt)
            for comprehension in node.generators:
                self.visit(comprehension)
            self.terminal(right)
            self.parent = gparent
        return visit

    visit_ListComp = generator_visit('[', ']')
    visit_GeneratorExp = generator_visit('(', ')')
    visit_SetComp = generator_visit('{', '}')
    del generator_visit

    def visit_DictComp(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal('{')
        self.visit(node.key)
        self.terminal(': ')
        self.visit(node.value)
        for comprehension in node.generators:
            self.visit(comprehension)
        self.terminal('}')
        self.parent = gparent

    def visit_IfExp(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.visit(node.body)
        self.terminal('if')
        self.visit(node.test)
        self.terminal('else')
        self.visit(node.orelse)
        self.parent = gparent

    def visit_Starred(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal('*')
        self.visit(node.value)
        self.parent = gparent

    def visit_Repr(self, node):
        gparent = self.parent
        self.non_terminal(node)
        # XXX: python 2.6 only
        self.terminal('`')
        self.visit(node.value)
        self.terminal('`')
        self.parent = gparent

    # Helper Nodes
    def visit_arg(self, node):
        self.terminal(node.arg)

    def visit_alias(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal(node.name)
        if node.asname is not None:
            self.terminal('as')
            self.terminal(node.asname)
        self.parent = gparent

    def visit_comprehension(self, node):
        gparent = self.parent
        self.non_terminal(node)
        self.terminal('for')
        self.visit(node.target)
        self.terminal(' in ')
        self.visit(node.iter)
        if node.ifs:
            for if_ in node.ifs:
                self.terminal(' if ')
                self.visit(if_)
        self.parent = gparent

    def visit_excepthandler(self, node):
        gparent = self.parent
        self.non_terminal(node)
        
        self.terminal('except')
        if node.type is not None:
            self.visit(node.type)
            if node.name is not None:
                self.terminal(' as ')
                self.visit(node.name)
        self.terminal(':')
        self.body(node.body)
        self.parent = gparent

if __name__ == "__main__":
    example = """
x = 0
y = f(x)
"""
    plot_code_graph(example)
