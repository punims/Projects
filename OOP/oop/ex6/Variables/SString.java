package oop.ex6.Variables;
import oop.ex6.Block.*;
import java.util.regex.Pattern;


public class SString extends Variable {

    final public static String STRING_VALUE = "[\\s]*\".*\"|[\\s]*"+ VARIABLE_NAME;

    /**
     * default constructor
     * @param name
     * @param isFinal
     * @param assignment
     */
    public SString(String name, boolean isFinal, boolean assignment, Block currentBlock) {
        super(name, isFinal, assignment, currentBlock);    }

    @Override
    boolean checkValue(String value) {
        p = Pattern.compile(STRING_VALUE);
        m = p.matcher(value);
        return m.matches();
    }
}
