package java2graph;

import java.util.ArrayList;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class MethodFinder extends VoidVisitorAdapter<Void> {

    public ArrayList<MethodDeclaration> allDeclarations = new ArrayList<>();

    @Override
    public void visit(MethodDeclaration n, Void arg) {
        allDeclarations.add(n);
        super.visit(n, arg);
    }

}