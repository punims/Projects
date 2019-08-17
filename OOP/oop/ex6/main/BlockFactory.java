package oop.ex6.main;

import oop.ex6.Block.*;
import oop.ex6.Block.Method;
import oop.ex6.Exceptions.BlockException;
import oop.ex6.Exceptions.VariableAlreadyExists;
import oop.ex6.Exceptions.VariableException;
import oop.ex6.Variables.Variable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class BlockFactory {

    static final String IF = "if", WHILE = "while", METHOD = "method";
    static final int CONDITION_GROUP = 2;
    static final int GROUP_NAME = 1, GROUP_PARAMETERS = 2, FINAL_NUM = 1, VAR_TYPE = 2,
            VAR_NAME = 3, PARAMETERS_CELL = 1, METHOD_NAME = 0;
    static final String RANDOM_STRING = "\"random\"", RANDOM_INT = "5", RANDOM_CHAR = "'a'";
    final static int STRING_NUM = 11, INT_NUUM = 15, CHAR_NUM = 12, BOOLEAN_NUM = 14, DOUBLE_NUM = 13;

    ///////////////////
    // REGEX
    ///////////////////
    static final String METHOD_DECLARATION = "^[\\s]*void[\\s]+([A-Za-z][\\w]*)[\\s]*\\(*([^)]*)\\)[\\s]*\\{[\\s]*$";
    static final String IF_WHILE = "^[\\s]*(if|while)[\\s]*\\(([^)]*)\\)[\\s]*\\{[\\s]*$";
    static final String CONDITION_PARAMETER = "(-)?([\\d]+\\.[\\d]+|[\\d]+)|true|false|_[\\w]+|[A-Za-z][\\w]*";
    static final String CONDITION = "^[\\s]*((" + CONDITION_PARAMETER + ")[\\s]*(\\|\\||&&)[\\s]*)*("
            + CONDITION_PARAMETER + ")[\\s]*$";
    static final String TRUE = "true", FALSE = "false";
    static final String VARIABLE_NAME = "_[\\w]+|[A-Za-z][\\w]*";
    static final String FINAL = "[\\s]*(final[\\s]+)?";
    static final String INT = "int", DOUBLE = "double", STRING = "String", CHAR = "char", BOOLEAN = "boolean";
    static final String TYPES = INT + "|" + CHAR + "|" + DOUBLE + "|" + STRING + "|" + BOOLEAN;
    static final String PARAMETER = "[\\s]*(final[\\s]+)?(int|String|boolean|double|char)" +
            "[\\s]+(_[\\w]+|[A-Za-z]+[\\w]*)[\\s]*";
    static final String PARAMETERS = "([\\s]*((final[\\s]+)?(int|String|boolean|double|char)" +
            "[\\s]+(_[\\w]+|[A-Za-z]+[\\w]*)[\\s]*,[\\s]*)*([\\s]*(final[\\s]+)?(int|String|boolean|double|char)" +
            "[\\s]+(_[\\w]+|[A-Za-z]+[\\w]*)))[\\s]*$";


    static private Pattern p;
    static private Matcher m;

    /**
     *
     * @param parent
     * @param line
     * @return true if the if while creation was properly with a valid condition. false otherwise.
     * @throws BlockException
     */
    public static Block createIfWhile(Block parent, String line) throws BlockException {
        p = Pattern.compile(IF_WHILE);
        m = p.matcher(line);
        if (m.matches()) {
            if (isCondition(m.group(CONDITION_GROUP), parent))
                return new IfWhile(parent);
        }
        throw new BlockException();
    }

    /**
     * Validate if the condition was given
     *
     * @param condition
     * @param parent
     * @return
     */
    private static boolean isCondition(String condition, Block parent) {
        p = Pattern.compile(CONDITION);
        m = p.matcher(condition);
        if (m.matches()) {
            p = Pattern.compile(CONDITION_PARAMETER);
            m = p.matcher(condition);
            while (m.find()) {
                if (!isValidBooleanParameter(m.group(), parent))
                    return false;
            }
            return true;
        }
        return false;
    }

    /**
     *  checks if a boolean parameter for an if/while block is valid.
     * @param parameter
     * @param parent
     * @return
     */
    private static boolean isValidBooleanParameter(String parameter, Block parent) {
        if (!parameter.matches(TRUE) && !parameter.matches(FALSE) && parameter.matches(VARIABLE_NAME)) {
            if (parent.globalContains(parameter)) {
                String paramType = parent.getVariableType(parameter);
                Variable variable = parent.getGlobalVariable(parameter);
                return (Variable.assignable(Variable.S_BOOLEAN, paramType) && variable.isAssigned());
            } else return false;
        }
        return true;
    }

    /**
     * a method creating function, creates local variables on the way.
     * @param line
     * @param global
     * @return
     * @throws BlockException
     */
    public static boolean createMethod(String line, Global global) throws BlockException, VariableException {
        String[] separated = getMethodGroups(line);
        if (!separated[PARAMETERS_CELL].equals("")) {
            p = Pattern.compile(PARAMETERS);
            m = p.matcher(separated[PARAMETERS_CELL]);
            if (m.matches()) {
                ArrayList<Variable> parameters = createParameters(separated[PARAMETERS_CELL], global);
                return global.addMethod(new Method(separated[METHOD_NAME], global, parameters));
            } else return false;
        } else if (!global.contains(separated[METHOD_NAME]))
            return global.addMethod(new Method(separated[METHOD_NAME], global, new ArrayList<>()));
        return false;
    }

    /**
     * a parser using regex.
     * @param line
     * @return
     */
    private static String[] getMethodGroups(String line) {
        p = Pattern.compile(METHOD_DECLARATION);
        m = p.matcher(line);
        m.matches();
        String[] declaration = {m.group(GROUP_NAME), m.group(GROUP_PARAMETERS)};
        return declaration;
    }


    /**
     * a helping method which creates parameters while creating a method object.
     * @param parameters
     * @param global
     * @return
     * @throws BlockException
     */
    private static ArrayList<Variable> createParameters(String parameters, Global global) throws VariableException{
        p = Pattern.compile(PARAMETER);
        m = p.matcher(parameters);
        Variable variable;
        ArrayList<Variable> params = new ArrayList<>();
        HashSet<String> names = new HashSet<>();
        while (m.find()) {
            switch (m.group(VAR_TYPE)) {
                case STRING:
                    variable = VariableFactory.createVariable(STRING_NUM, m.group(VAR_NAME), RANDOM_STRING, m
                            .group(FINAL_NUM) != null, global);
                    break;
                case CHAR:
                    variable = VariableFactory.createVariable(CHAR_NUM, m.group(VAR_NAME), RANDOM_CHAR, m.group
                            (FINAL_NUM) != null, global);
                    break;
                case DOUBLE:
                    variable = VariableFactory.createVariable(DOUBLE_NUM, m.group(VAR_NAME), RANDOM_INT, m.group
                            (FINAL_NUM) != null, global);
                    break;
                case BOOLEAN:
                    variable = VariableFactory.createVariable(BOOLEAN_NUM, m.group(VAR_NAME), RANDOM_INT, m
                            .group(FINAL_NUM) != null, global);
                    break;
                default:
                    variable = VariableFactory.createVariable(INT_NUUM, m.group(VAR_NAME), RANDOM_INT, m.group
                            (FINAL_NUM) != null, global);
            }
            if (names.contains(variable.getName()))
                throw new VariableAlreadyExists();
            params.add(variable);
            names.add(variable.getName());
        }
        return params;
    }

    /**
     * another regex parser.
     * @param line
     * @return
     */
    static String[] getMethodCallGroups(String line) {
        p = Pattern.compile(FirstScan.METHOD_CALL);
        m = p.matcher(line);
        m.matches();
        String[] declaration = {m.group(GROUP_NAME), m.group(GROUP_PARAMETERS)};
        return declaration;
    }
}