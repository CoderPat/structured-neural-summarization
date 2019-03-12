package java2graph;

import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.Stack;
import java.util.Map.Entry;

import com.github.javaparser.JavaToken;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.ForeachStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.visitor.VoidVisitorWithDefaults;
import com.github.javaparser.resolution.declarations.ResolvedValueDeclaration;
import com.github.javaparser.symbolsolver.model.resolution.SymbolReference;
import com.github.javaparser.symbolsolver.model.resolution.TypeSolver;
import com.github.javaparser.symbolsolver.resolution.SymbolSolver;

class DataflowGraphExtractor extends VoidVisitorWithDefaults<Void>{

    public static class ResolvedValueDeclarationWrapper {
        public final String name;
        public final Class<? extends ResolvedValueDeclaration> symbolClass;

        public static ResolvedValueDeclarationWrapper Of(ResolvedValueDeclaration declaration) {
            return new ResolvedValueDeclarationWrapper(declaration);
        }

        public static ResolvedValueDeclarationWrapper Of(String name) {
            return new ResolvedValueDeclarationWrapper(name);
        }

        private ResolvedValueDeclarationWrapper(ResolvedValueDeclaration declaration) {
            this.name = declaration.getName();
            this.symbolClass = declaration.getClass();
        }

        private ResolvedValueDeclarationWrapper(String name) {
            this.name = name;
            this.symbolClass = null;
        }

        @Override
        public int hashCode() {
            return name.hashCode();
        }

        @Override
        public boolean equals(Object other) {
            if (other instanceof ResolvedValueDeclarationWrapper) {
                String otherDeclname = ((ResolvedValueDeclarationWrapper)other).name;
                Class<? extends ResolvedValueDeclaration> otherSymbolClass = ((ResolvedValueDeclarationWrapper)other).symbolClass;
                if (!Objects.equals(name, otherDeclname)
                    || (this.symbolClass != null && !Objects.equals(symbolClass, otherSymbolClass))) {
                        return false;
                    }
                return true;
            }
            return false;
        }
    }

    public static final String LAST_USE = "LastUse";

    private Graph<Object> graph;
    private TypeSolver typeSolver;

    private Stack<Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>>> lastUsedNode = new Stack<>();

    public DataflowGraphExtractor(Graph<Object> graph, TypeSolver typeSolver) {
        this.graph = graph;
        this.typeSolver = typeSolver;
    }

    @Override
    public void defaultAction(Node node, Void v) {
        for(Node child: node.getChildNodes()) {
            child.accept(this, null);
        }
    }

    @Override
    public void defaultAction(NodeList nodeList, Void arg) {
        for(Object child: nodeList) {
            Node c = (Node) child;
            c.accept(this, null);
        }
    }

    private static Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> cloneState(
        Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> state) {
        Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> copy = new HashMap<>();
        for(Entry<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> entry : state.entrySet()) {
            IdentityHashMap<JavaToken, Void> incomingValues = new IdentityHashMap<>();
            for(JavaToken t : entry.getValue().keySet()) {
                incomingValues.put(t, null);
            }
            copy.put(entry.getKey(), incomingValues);
        }
        return copy;
    }

    private static void mergeStatesIntoFirst(Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> first,
        Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> other) {
        for(Entry<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> otherEntry: other.entrySet()) {
            if (!first.containsKey(otherEntry.getKey())) {
                first.put(otherEntry.getKey(), otherEntry.getValue());
            } else {
                first.get(otherEntry.getKey()).putAll(otherEntry.getValue());
            }
        }
    }

    @Override
    public void visit(final SimpleName n, final Void arg) {
        SymbolSolver ss = new SymbolSolver(typeSolver);
        
        SymbolReference<? extends ResolvedValueDeclaration> ref = null;

        Class<? extends Node> parentClass = n.getParentNode().get().getClass();
        if (parentClass.equals(ClassOrInterfaceDeclaration.class)) {                
            return;
        }
        
        try {
            ref = ss.solveSymbol(n.asString(), n);            
        } catch (Exception e) {
            // be silent
        }
        
        ResolvedValueDeclarationWrapper declarationSymbol;
        if (ref != null && ref.isSolved()) {
            declarationSymbol = ResolvedValueDeclarationWrapper.Of(ref.getCorrespondingDeclaration());
        } else {
            declarationSymbol = ResolvedValueDeclarationWrapper.Of(n.getId()); // Fallback to name only
        }
           
        if (!lastUsedNode.peek().containsKey(declarationSymbol)) {
            lastUsedNode.peek().put(declarationSymbol, new IdentityHashMap<>());
        }
        IdentityHashMap<JavaToken, Void> incomingEdgesFrom = lastUsedNode.peek().get(declarationSymbol);
        JavaToken currentToken = n.getTokenRange().get().getBegin();
        for(JavaToken incomingToken: incomingEdgesFrom.keySet()) {
            graph.addEdge(incomingToken, currentToken, LAST_USE);
        }
        
        incomingEdgesFrom.clear();
        incomingEdgesFrom.put(currentToken, null);
        
    }

    @Override
    public void visit(final ClassOrInterfaceDeclaration n, final Void arg) {
        lastUsedNode.push(new HashMap<>());
        defaultAction(n, arg);
        lastUsedNode.pop();
    }

    @Override
    public void visit(final MethodDeclaration n, final Void arg) {
        lastUsedNode.push(new HashMap<>());
        defaultAction(n, arg);
        lastUsedNode.pop();
    }

    @Override
    public void visit(final IfStmt n, final Void arg) {
        n.getCondition().accept(this, null);
        Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> branchState = cloneState(lastUsedNode.peek());
        
        n.getThenStmt().accept(this, null);
        Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> postThenState = lastUsedNode.pop();
        lastUsedNode.push(branchState);

        if (n.hasElseBranch()) {
            n.getElseStmt().get().accept(this, null);
        }
        mergeStatesIntoFirst(lastUsedNode.peek(), postThenState);
    }

    @Override
    public void visit(final TryStmt n, final Void arg) {
        n.getTryBlock().accept(this, null);

        Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> beforeState = cloneState(lastUsedNode.peek());
        Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> postCatchState = cloneState(lastUsedNode.peek());
        lastUsedNode.pop();
        for(Node catchClause: n.getCatchClauses()) {
            lastUsedNode.push(cloneState(beforeState));
            catchClause.accept(this, null);
            mergeStatesIntoFirst(postCatchState, lastUsedNode.pop());
        }
        lastUsedNode.push(postCatchState);

        if (n.getFinallyBlock().isPresent()) {
            n.getFinallyBlock().get().accept(this, null);
        }
    }

    @Override
    public void visit(final WhileStmt n, final Void arg) {
        // TODO: Handle break, continue
        n.getCondition().accept(this, null);
        Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> noBodyExecutionState = cloneState(lastUsedNode.peek());
        n.getBody().accept(this, null);
        n.getBody().accept(this, null); // Unfold twice
        mergeStatesIntoFirst(lastUsedNode.peek(), noBodyExecutionState);
    }

    @Override
    public void visit(final ForStmt n, final Void arg) {
        // TODO: Handle break, continue
        for(Node init : n.getInitialization()) init.accept(this, null);
        if (n.getCompare().isPresent()) {
            n.getCompare().get().accept(this, null);
        }

        Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> noBodyExecutionState = cloneState(lastUsedNode.peek());
        for (int i = 0; i < 2; i++)  {  // Unfold twice
            n.getBody().accept(this, null);
            for (Node updater: n.getUpdate()) {
                updater.accept(this, null);
            }
            if (n.getCompare().isPresent()) {
                n.getCompare().get().accept(this, null);
            }
        }

        mergeStatesIntoFirst(lastUsedNode.peek(), noBodyExecutionState);
    }

    @Override
    public void visit(final ForeachStmt n, final Void arg) {
        // TODO: Handle break, continue
        n.getIterable().accept(this, null);
        Map<ResolvedValueDeclarationWrapper, IdentityHashMap<JavaToken, Void>> noBodyExecutionState = cloneState(lastUsedNode.peek());

        for (int i = 0; i < 2; i++)  {  // Unfold twice
            n.getVariable().accept(this, null);
            n.getBody().accept(this, null);
        }

        mergeStatesIntoFirst(lastUsedNode.peek(), noBodyExecutionState);
    }
}