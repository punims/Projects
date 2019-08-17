package oop.ex6.Block;
import oop.ex6.Variables.Variable;
import java.util.ArrayList;
import java.util.HashMap;

public class Global extends Block{

    private static Global global = null;
    private HashMap<String,Method> methods;

    /**
     * Create a Global object using Singleton principle.
     * @return Global object.
     */
    public static Global getGlobal(){
        if (global == null)
            global = new Global();
        return global;
    }

    public static void reset() {
        global = new Global();
    }

    /**
     * default constructor of global object.
     */
    private Global(){
        super();
        this.methods = new HashMap<>();
    }

    /**
     *
     * @param method an instance of Method object.
     * @return true if the method was added successfully and false otherwise.
     */
    public boolean addMethod(Method method){
        if (!contains(method)) {
            methods.put(method.getName(), method);
            return true;
        } else
            return false;
    }

    /**
     * getter function which gets a method according to its' String name.
     * @return the given Method.
     */

    public Method getMethod(String methodName){
        return methods.get(methodName);

    }

    /**
     *
     * @param method an instance of Method object.
     * @return true if the exists and false otherwise.
     */
    public boolean contains(Method method){
        return this.methods.containsKey(method.getName());
    }

    /**
     * overriden getter with a string name.
     * @param methodName
     * @return
     */
    public boolean contains(String methodName){
        return this.methods.containsKey(methodName);
    }

}
