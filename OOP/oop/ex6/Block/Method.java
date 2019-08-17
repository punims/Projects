package oop.ex6.Block;


import oop.ex6.Variables.Variable;

import java.util.ArrayList;

public class Method extends Block {

    private String name;
    private ArrayList<Variable> parameters;

    /**
     * Constructor for a Method instance.
     *
     * @param name
     * @param parent
     */
    public Method(String name, Block parent, ArrayList<Variable> parameters) {
        super(parent);
        this.name = name;
        this.parameters = parameters;
        for (Variable variable : parameters){
            addLocalVariable(variable);
        }
    }

    public Method(String name, Block parent) {
        super(parent);
        this.name = name;
        this.parameters = new ArrayList<>();
    }

    /**
     * @return a string represent a method name.
     */
    public String getName() {
        return this.name;
    }


    /**
     * getter function for the amount of parameters in the Method.
     * @return the size of the Hashmap.
     */
    public int getParameterSize(){
        return parameters.size();
    }

    /**
     * a getter function for any variable in the method according to the index.
     * @param index
     * @return
     */
    public Variable getParameters(int index){
        return parameters.get(index);
    }

}
