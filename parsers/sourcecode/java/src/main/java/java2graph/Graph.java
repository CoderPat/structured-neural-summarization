package java2graph;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.function.Function;

import com.github.javaparser.utils.StringEscapeUtils;

public class Graph<T> {
    protected List<T> nodeLabels = new ArrayList<T>();
    protected Map<T, Integer> nodeToIdx = new IdentityHashMap<T, Integer>();
    protected Map<String, Map<Integer, Set<Integer>>> edges = new HashMap<>();

    public int getNodeId(T node) {
        return nodeToIdx.computeIfAbsent(node, n -> {
            int idx = nodeLabels.size();
            nodeLabels.add(n);
            nodeToIdx.put(n, idx);
            return idx;
        });        
    }

    public boolean containsNode(T node) {
        return nodeToIdx.containsKey(node);
    }

    public void addEdge(T from, T to, String edgeType) {
        int fromIdx = getNodeId(from);
        int toIdx = getNodeId(to);
        if (!edges.containsKey(edgeType)) {
            edges.put(edgeType, new HashMap<>());
        }
        
        Map<Integer, Set<Integer>> outEdges = edges.get(edgeType);
        if(!outEdges.containsKey(fromIdx)) {
            outEdges.put(fromIdx, new HashSet<>());
        }

        Set<Integer> targetNodes = outEdges.get(fromIdx);
        targetNodes.add(toIdx);
    }

	public String toDot(Function<T, String> nodeLabeler) {
        StringBuffer sb = new StringBuffer();
		for(int i=0; i < nodeLabels.size(); i++) {
            sb.append(i + " [label=\"" + StringEscapeUtils.escapeJava(nodeLabeler.apply(nodeLabels.get(i))) + "\"];\n");
        }

        for(Entry<String, Map<Integer, Set<Integer>>> edgesOfType: edges.entrySet()) {
            for(Entry<Integer, Set<Integer>> outEdges: edgesOfType.getValue().entrySet()) {
                for(int targetNodeIdx: outEdges.getValue()) {
                    sb.append(outEdges.getKey() + "->" + targetNodeIdx + " [label=\"" + edgesOfType.getKey() + "\"];\n");
                }
            }
        }
        return sb.toString();
    }
    
    public static class JsonSerializableGraph{
        public Map<Integer, String> NodeLabels = new HashMap<Integer, String>();
        public Map<String, List<Integer[]>> Edges = new HashMap<>();
    }

    public JsonSerializableGraph toJsonSerializableObject(Function<T, String> nodeLabeler) throws IOException {
        JsonSerializableGraph graph = new JsonSerializableGraph();
        for (int i=0; i < this.nodeLabels.size(); i++) {
            graph.NodeLabels.put(i, nodeLabeler.apply(this.nodeLabels.get(i)));
        }

        for (Entry<String, Map<Integer, Set<Integer>>> edgesOfType : this.edges.entrySet()) {
            List<Integer[]> adjList = new ArrayList<>();
            graph.Edges.put(edgesOfType.getKey(), adjList);
            for (Entry<Integer, Set<Integer>> fromToEdge : edgesOfType.getValue().entrySet()) {
                for (int targetEdge: fromToEdge.getValue()) {
                    Integer[] edge = new Integer[] {fromToEdge.getKey(), targetEdge};
                    adjList.add(edge);
                }
            }
        }
        return graph;
    }
}