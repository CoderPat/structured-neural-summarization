package java2graph;

import java.util.HashMap;
import java.util.Map;

import com.github.javaparser.GeneratedJavaParserConstants;
import com.github.javaparser.JavaToken;
import com.github.javaparser.TokenRange;
import com.github.javaparser.ast.Node;
import com.github.javaparser.symbolsolver.model.resolution.TypeSolver;

public class JavaGraph {    

    public final static String NEXT_TOKEN = "NextToken";
    public final static String LAST_LEXICAL_USE = "LastLexicalUse";

    public static Graph<Object> CreateGraph(Node node, TypeSolver typeSolver) {
        Graph<Object> codeGraph = new Graph<Object>();

        // Add token-level info
        Map<String, JavaToken> lastLexicalUsages = new HashMap<>();

        TokenRange tokenRange =  node.getTokenRange().get();
        for (JavaToken token : tokenRange) {
            if (!token.getNextToken().isPresent() || IsExcludedKind(token)) {
                continue;
            }
            if (token.getKind() == GeneratedJavaParserConstants.IDENTIFIER) {
                if (lastLexicalUsages.containsKey(token.getText())) {
                    codeGraph.addEdge(token, lastLexicalUsages.get(token.getText()), LAST_LEXICAL_USE);
                }
                lastLexicalUsages.put(token.getText(), token);
            }

            JavaToken nextToken = token.getNextToken().get();
            while (IsExcludedKind(nextToken)) {
                if (!nextToken.getNextToken().isPresent()) {
                    nextToken = null;
                    break;
                }
                nextToken = nextToken.getNextToken().get();
            }
            if (nextToken == null) continue;
            codeGraph.addEdge(token, nextToken, NEXT_TOKEN);
        }

        // Now add AST-level info
        node.accept(new AstGraphCreator(codeGraph), null);
        // Now add data-flow info
        //node.accept(new DataflowGraphExtractor(codeGraph, typeSolver), null);
        return codeGraph;
    }

    private static boolean IsExcludedKind(JavaToken token) {
        switch(JavaToken.Kind.valueOf(token.getKind())) {
            case SPACE:
            case EOF:
            case UNIX_EOL:
            case WINDOWS_EOL:
            case OLD_MAC_EOL:
            
            case ENTER_JAVADOC_COMMENT:
            case ENTER_MULTILINE_COMMENT:
            case JAVADOC_COMMENT:
            case MULTI_LINE_COMMENT:
            case SINGLE_LINE_COMMENT:
            case COMMENT_CONTENT:
                return true;
            default:
                return false;
        }
    }
}