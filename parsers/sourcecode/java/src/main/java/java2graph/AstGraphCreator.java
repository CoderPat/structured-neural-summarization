package java2graph;

import java.util.IdentityHashMap;
import java.util.Map;

import com.github.javaparser.JavaToken;
import com.github.javaparser.TokenRange;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.visitor.VoidVisitorWithDefaults;

public class AstGraphCreator extends VoidVisitorWithDefaults<Void> {

    public static final String CHILD_EDGE = "Child";

    private Graph<Object> graph;

	public AstGraphCreator(Graph<Object> graph) {
        this.graph = graph;
    }

    @Override
    public void defaultAction(Node node, Void v) {
        for(Node child: node.getChildNodes()) {
            graph.addEdge(node, child, CHILD_EDGE);
            child.accept(this, null);
        }
        AddAllTokensNotInChildren(node);
    }

    @Override
    public void defaultAction(NodeList nodeList, Void arg) {
        for(Object child: nodeList) {
            Node c = (Node) child;
            graph.addEdge(nodeList.getParentNode().get(), c, CHILD_EDGE);
            c.accept(this, null);
        }
    }

    private void AddAllTokensNotInChildren(Node node) {
        if (!node.getTokenRange().isPresent()) return;
        TokenRange tokensUnderNode = node.getTokenRange().get();
        Map<JavaToken, Void> tokens = new IdentityHashMap<>();
        for(JavaToken t: tokensUnderNode) {
            tokens.put(t, null);
        }
        
        for(Node child: node.getChildNodes()) {
            if (!child.getTokenRange().isPresent()) continue;
            for(JavaToken t: child.getTokenRange().get()) {
                tokens.remove(t);
            }
        }

        for(JavaToken t : tokens.keySet()) {
            if (graph.containsNode(t)) {
                graph.addEdge(node, t, CHILD_EDGE);
            }
        }
    }

    @Override
    public void visit(final SimpleName n, final Void arg) {
        // Compress graph, by _not_ including the SimpleName node
        graph.addEdge(n.getParentNode().get(), n.getTokenRange().get().getBegin(), CHILD_EDGE);
    }
}