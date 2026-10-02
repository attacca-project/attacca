import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.jar.JarFile;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldInsnNode;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.InsnNode;
import org.objectweb.asm.tree.IntInsnNode;
import org.objectweb.asm.tree.LdcInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.VarInsnNode;

public final class RendererMaskJarPatcher {
    private static final String RECORDER_ENTRY =
        "com/minerl/multiagent/recorder/PlayRecorder.class";
    private static final String RECORDER_OWNER =
        "com/minerl/multiagent/recorder/PlayRecorder";
    private static final String RENDERER_ENTRY =
        "net/minecraft/client/renderer/GameRenderer.class";
    private static final String RENDERER_OWNER =
        "net/minecraft/client/renderer/GameRenderer";
    private static final String HELPER =
        "com/minerl/multiagent/recorder/XBenchViewmodelCapture";

    private RendererMaskJarPatcher() {}

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException("usage: input.jar output-root");
        }
        try (JarFile jar = new JarFile(args[0])) {
            writePatchedRecorder(readAll(jar, RECORDER_ENTRY), Paths.get(args[1]));
            writePatchedRenderer(readAll(jar, RENDERER_ENTRY), Paths.get(args[1]));
        }
    }

    private static byte[] readAll(JarFile jar, String name) throws IOException {
        if (jar.getJarEntry(name) == null) {
            throw new IOException("jar lacks " + name);
        }
        try (InputStream input = jar.getInputStream(jar.getJarEntry(name));
                ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            final byte[] buffer = new byte[16384];
            for (int count; (count = input.read(buffer)) >= 0;) {
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private static void writeClass(ClassNode node, Path root) throws IOException {
        final ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS);
        node.accept(writer);
        final Path output = root.resolve(node.name + ".class");
        Files.createDirectories(output.getParent());
        Files.write(output, writer.toByteArray());
    }

    private static void writePatchedRecorder(byte[] original, Path root)
            throws IOException {
        final ClassNode node = new ClassNode(Opcodes.ASM7);
        new ClassReader(original).accept(node, 0);
        if (!RECORDER_OWNER.equals(node.name)) {
            throw new IllegalStateException("unexpected PlayRecorder owner " + node.name);
        }
        patchStartAllocation(node.methods);
        replaceCaptureBody(node.methods);
        writeClass(node, root);
    }

    private static void writePatchedRenderer(byte[] original, Path root)
            throws IOException {
        final ClassNode node = new ClassNode(Opcodes.ASM7);
        new ClassReader(original).accept(node, 0);
        if (!RENDERER_OWNER.equals(node.name)) {
            throw new IllegalStateException("unexpected GameRenderer owner " + node.name);
        }
        final MethodNode method = exactlyOne(
            node.methods, "renderHand",
            "(Lcom/mojang/blaze3d/matrix/MatrixStack;"
            + "Lnet/minecraft/client/renderer/ActiveRenderInfo;F)V");
        final InsnList body = new InsnList();
        body.add(new VarInsnNode(Opcodes.ALOAD, 0));
        body.add(new VarInsnNode(Opcodes.ALOAD, 1));
        body.add(new VarInsnNode(Opcodes.ALOAD, 2));
        body.add(new VarInsnNode(Opcodes.FLOAD, 3));
        body.add(new MethodInsnNode(
            Opcodes.INVOKESTATIC, HELPER, "renderHandWithMask",
            "(Lnet/minecraft/client/renderer/GameRenderer;"
            + "Lcom/mojang/blaze3d/matrix/MatrixStack;"
            + "Lnet/minecraft/client/renderer/ActiveRenderInfo;F)V", false));
        body.add(new InsnNode(Opcodes.RETURN));
        method.instructions.clear();
        method.tryCatchBlocks.clear();
        method.localVariables = null;
        method.instructions.add(body);
        patchWorldDepthCapture(node.methods);
        writeClass(node, root);
    }

    private static void patchWorldDepthCapture(List<MethodNode> methods) {
        final MethodNode world = exactlyOne(
            methods, "renderWorld",
            "(FJLcom/mojang/blaze3d/matrix/MatrixStack;)V");

        AbstractInsnNode handCall = null;
        for (AbstractInsnNode insn = world.instructions.getFirst();
                insn != null; insn = insn.getNext()) {
            if (insn instanceof MethodInsnNode) {
                final MethodInsnNode call = (MethodInsnNode) insn;
                if (RENDERER_OWNER.equals(call.owner)
                        && "renderHand".equals(call.name)
                        && ("(Lcom/mojang/blaze3d/matrix/MatrixStack;"
                            + "Lnet/minecraft/client/renderer/ActiveRenderInfo;F)V")
                           .equals(call.desc)) {
                    handCall = insn;
                    break;
                }
            }
        }
        if (handCall == null) {
            throw new IllegalStateException(
                "renderWorld lacks the private renderHand call");
        }

        AbstractInsnNode clearCall = null;
        for (AbstractInsnNode insn = handCall.getPrevious();
                insn != null; insn = insn.getPrevious()) {
            if (insn instanceof MethodInsnNode) {
                final MethodInsnNode call = (MethodInsnNode) insn;
                if ("com/mojang/blaze3d/systems/RenderSystem".equals(call.owner)
                        && "clear".equals(call.name)) {
                    clearCall = insn;
                    break;
                }
            }
        }
        if (clearCall == null) {
            throw new IllegalStateException(
                "renderWorld lacks the pre-hand depth clear");
        }

        AbstractInsnNode maskPush = null;
        for (AbstractInsnNode insn = clearCall.getPrevious();
                insn != null; insn = insn.getPrevious()) {
            if (insn instanceof IntInsnNode
                    && ((IntInsnNode) insn).operand == 256) {
                maskPush = insn;
                break;
            }
            if (insn instanceof LdcInsnNode
                    && Integer.valueOf(256).equals(((LdcInsnNode) insn).cst)) {
                maskPush = insn;
                break;
            }
        }
        if (maskPush == null) {
            throw new IllegalStateException(
                "renderWorld depth-clear mask (256) not found");
        }
        world.instructions.insertBefore(maskPush, new MethodInsnNode(
            Opcodes.INVOKESTATIC, HELPER, "captureWorldDepth", "()V", false));
        final InsnList entityIds = new InsnList();

        entityIds.add(new VarInsnNode(Opcodes.ALOAD, 4));
        entityIds.add(new VarInsnNode(Opcodes.FLOAD, 1));
        entityIds.add(new MethodInsnNode(
            Opcodes.INVOKESTATIC, HELPER, "captureEntityIds",
            "(Lcom/mojang/blaze3d/matrix/MatrixStack;F)V", false));
        world.instructions.insertBefore(maskPush, entityIds);
    }

    private static MethodNode exactlyOne(
            List<MethodNode> methods, String name, String descriptor) {
        MethodNode found = null;
        for (MethodNode method : methods) {
            if (name.equals(method.name) && descriptor.equals(method.desc)) {
                if (found != null) {
                    throw new IllegalStateException("duplicate method " + name);
                }
                found = method;
            }
        }
        if (found == null) {
            throw new IllegalStateException("missing method " + name + descriptor);
        }
        return found;
    }

    private static void patchStartAllocation(List<MethodNode> methods) {
        final MethodNode start = exactlyOne(methods, "start", "()V");
        int replacements = 0;
        for (AbstractInsnNode insn = start.instructions.getFirst();
                insn != null; insn = insn.getNext()) {
            if (!(insn instanceof MethodInsnNode)) {
                continue;
            }
            final MethodInsnNode call = (MethodInsnNode) insn;
            if (!"java/nio/ByteBuffer".equals(call.owner)
                    || !"allocateDirect".equals(call.name)
                    || !"(I)Ljava/nio/ByteBuffer;".equals(call.desc)) {
                continue;
            }
            AbstractInsnNode cursor = insn.getPrevious();
            for (int distance = 0; cursor != null && distance < 8;
                    ++distance, cursor = cursor.getPrevious()) {
                if (cursor.getOpcode() == Opcodes.ICONST_3) {

                    start.instructions.set(
                        cursor, new IntInsnNode(Opcodes.BIPUSH, 20));
                    ++replacements;
                    break;
                }
            }
        }
        if (replacements != 1) {
            throw new IllegalStateException(
                "expected one PlayRecorder image buffer multiplier, found "
                + replacements);
        }
    }

    private static void replaceCaptureBody(List<MethodNode> methods) {
        final MethodNode method = exactlyOne(
            methods, "getRGBFrame", "(Ljava/nio/ByteBuffer;)V");
        final InsnList body = new InsnList();
        body.add(new VarInsnNode(Opcodes.ALOAD, 0));
        body.add(new FieldInsnNode(
            Opcodes.GETFIELD, RECORDER_OWNER, "mc",
            "Lnet/minecraft/client/Minecraft;"));
        body.add(new VarInsnNode(Opcodes.ALOAD, 0));
        body.add(new FieldInsnNode(
            Opcodes.GETFIELD, RECORDER_OWNER, "width", "I"));
        body.add(new VarInsnNode(Opcodes.ALOAD, 0));
        body.add(new FieldInsnNode(
            Opcodes.GETFIELD, RECORDER_OWNER, "height", "I"));
        body.add(new VarInsnNode(Opcodes.ALOAD, 1));
        body.add(new MethodInsnNode(
            Opcodes.INVOKESTATIC, HELPER, "capture",
            "(Lnet/minecraft/client/Minecraft;IILjava/nio/ByteBuffer;)V", false));
        body.add(new InsnNode(Opcodes.RETURN));
        method.instructions.clear();
        method.tryCatchBlocks.clear();
        method.localVariables = null;
        method.instructions.add(body);
    }
}
