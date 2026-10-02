package com.minerl.multiagent.recorder;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.DefaultColorVertexBuilder;
import com.mojang.blaze3d.vertex.IVertexBuilder;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.BufferBuilder;
import net.minecraft.client.renderer.ActiveRenderInfo;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.IRenderTypeBuffer;
import net.minecraft.client.renderer.RenderState;
import net.minecraft.client.renderer.RenderType;
import net.minecraft.client.renderer.entity.EntityRenderer;
import net.minecraft.client.renderer.entity.EntityRendererManager;
import net.minecraft.client.renderer.vertex.DefaultVertexFormats;
import net.minecraft.client.shader.Framebuffer;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityType;
import net.minecraft.entity.passive.SheepEntity;
import net.minecraft.entity.passive.WolfEntity;
import net.minecraft.entity.passive.horse.LlamaEntity;
import net.minecraft.item.DyeColor;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.ResourceLocation;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.vector.Vector3d;
import com.mojang.blaze3d.matrix.MatrixStack;
import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL30;

public final class XBenchViewmodelCapture {
    private static Framebuffer maskFramebuffer;
    private static ByteBuffer latestMaskAlpha;
    private static int latestMaskWidth;
    private static int latestMaskHeight;
    private static FloatBuffer latestDepthFloats;
    private static int latestDepthWidth;
    private static int latestDepthHeight;
    private static Framebuffer entityIdFramebuffer;
    private static ByteBuffer latestEntityIdRgb;
    private static int latestEntityIdWidth;
    private static int latestEntityIdHeight;
    private static boolean latestEntityIdValid;
    private static int latestSceneAliveClassesMask;
    private static float latestUniqueWolfHealth = Float.NaN;
    private static boolean entityIdErrorReported;
    private static volatile String snapshotSaveInFlight;

    private static final int SEMANTIC_SHEEP = 1;
    private static final int SEMANTIC_COW = 2;
    private static final int SEMANTIC_PIG = 3;
    private static final int SEMANTIC_CHICKEN = 4;
    private static final int SEMANTIC_HORSE = 5;
    private static final int SEMANTIC_PANDA = 6;
    private static final int SEMANTIC_MOOSHROOM = 7;
    private static final int SEMANTIC_BLACK_SHEEP = 8;
    private static final int SEMANTIC_WHITE_SHEEP = 9;
    private static final int SEMANTIC_GRAY_SHEEP = 10;
    private static final int SEMANTIC_LIGHT_GRAY_SHEEP = 11;
    private static final int SEMANTIC_BROWN_SHEEP = 12;
    private static final int SEMANTIC_BLUE_SHEEP = 13;
    private static final int SEMANTIC_PURPLE_SHEEP = 14;
    private static final int SEMANTIC_CREAMY_TRADER_LLAMA = 15;
    private static final int SEMANTIC_WHITE_TRADER_LLAMA = 16;
    private static final int SEMANTIC_RED_SHEEP = 17;
    private static final int SEMANTIC_YELLOW_SHEEP = 18;
    private static final int SEMANTIC_ORANGE_SHEEP = 19;
    private static final int SEMANTIC_TURTLE = 20;
    private static final int SEMANTIC_LLAMA = 21;
    private static final int SEMANTIC_DONKEY = 22;
    private static final int SEMANTIC_FOX = 23;
    private static final int SEMANTIC_CAT = 24;
    private static final int SEMANTIC_OCELOT = 25;
    private static final int SEMANTIC_POLAR_BEAR = 26;
    private static final int SEMANTIC_RABBIT = 27;
    private static final int SEMANTIC_PARROT = 28;
    private static final int SEMANTIC_WOLF = 29;
    private static final int MAX_TRANSPORT_ENTITY_ID = 0xffff;

    private static final Path SNAPSHOT_SAVE_REQUEST =
        Paths.get(".xbench_snapshot_save_request");
    private static final Path SNAPSHOT_SAVE_COMPLETE =
        Paths.get(".xbench_snapshot_save_complete");

    private XBenchViewmodelCapture() {}

    public static void renderHandWithMask(
            GameRenderer renderer, MatrixStack matrix,
            ActiveRenderInfo camera, float partialTicks) {
        final Minecraft minecraft = Minecraft.getInstance();
        final Framebuffer main = minecraft.getFramebuffer();

        renderer.renderHand(matrix, camera, partialTicks, true, true, false);

        final int width = main.framebufferWidth;
        final int height = main.framebufferHeight;
        if (maskFramebuffer == null
                || maskFramebuffer.framebufferWidth != width
                || maskFramebuffer.framebufferHeight != height) {
            if (maskFramebuffer != null) {
                maskFramebuffer.deleteFramebuffer();
            }
            maskFramebuffer = new Framebuffer(
                width, height, true, Minecraft.IS_RUNNING_ON_MAC);
            maskFramebuffer.setFramebufferColor(0.0f, 0.0f, 0.0f, 0.0f);
        }

        maskFramebuffer.bindFramebuffer(true);
        maskFramebuffer.framebufferClear(Minecraft.IS_RUNNING_ON_MAC);
        renderer.renderHand(matrix, camera, partialTicks, true, true, false);
        final int pixels = Math.multiplyExact(width, height);
        if (latestMaskAlpha == null || latestMaskAlpha.capacity() != pixels) {
            latestMaskAlpha = BufferUtils.createByteBuffer(pixels);
        }
        latestMaskAlpha.clear();
        GL11.glReadPixels(
            0, 0, width, height,
            GL11.GL_ALPHA, GL11.GL_UNSIGNED_BYTE, latestMaskAlpha);
        latestMaskWidth = width;
        latestMaskHeight = height;
        main.bindFramebuffer(true);
    }

    public static void captureWorldDepth() {
        final Minecraft minecraft = Minecraft.getInstance();
        pollSnapshotSaveRequest(minecraft);
        final Framebuffer main = minecraft.getFramebuffer();
        main.bindFramebuffer(true);
        final int width = main.framebufferWidth;
        final int height = main.framebufferHeight;
        final int pixels = Math.multiplyExact(width, height);
        if (latestDepthFloats == null
                || latestDepthFloats.capacity() != pixels) {
            latestDepthFloats = BufferUtils.createFloatBuffer(pixels);
        }
        latestDepthFloats.clear();
        GL11.glReadPixels(
            0, 0, width, height,
            GL11.GL_DEPTH_COMPONENT, GL11.GL_FLOAT, latestDepthFloats);
        latestDepthWidth = width;
        latestDepthHeight = height;
    }

    public static void captureEntityIds(
            MatrixStack matrix, float partialTicks) {
        final Minecraft minecraft = Minecraft.getInstance();
        final Framebuffer main = minecraft.getFramebuffer();
        final int width = main.framebufferWidth;
        final int height = main.framebufferHeight;
        latestEntityIdValid = false;
        latestUniqueWolfHealth = Float.NaN;
        if (minecraft.world == null || minecraft.player == null) {
            return;
        }
        try {
            latestSceneAliveClassesMask = 0;
            ensureEntityIdFramebuffer(width, height);
            entityIdFramebuffer.bindFramebuffer(true);
            entityIdFramebuffer.framebufferClear(Minecraft.IS_RUNNING_ON_MAC);

            GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, main.framebufferObject);
            GL30.glBindFramebuffer(
                GL30.GL_DRAW_FRAMEBUFFER, entityIdFramebuffer.framebufferObject);
            GL30.glBlitFramebuffer(
                0, 0, width, height, 0, 0, width, height,
                GL11.GL_DEPTH_BUFFER_BIT, GL11.GL_NEAREST);
            entityIdFramebuffer.bindFramebuffer(true);

            final BufferBuilder builder = new BufferBuilder(256);
            final IRenderTypeBuffer.Impl sink = IRenderTypeBuffer.getImpl(builder);
            final EntityIdBuffer ids = new EntityIdBuffer(sink);
            final ActiveRenderInfo camera = minecraft.gameRenderer.getActiveRenderInfo();
            final Vector3d eye = camera.getProjectedView();
            final EntityRendererManager manager = minecraft.getRenderManager();
            int aliveWolfCount = 0;
            float aliveWolfHealth = Float.NaN;
            for (Entity entity : minecraft.world.getAllEntities()) {
                final int semantic = semanticCode(entity);
                if (semantic == SEMANTIC_WOLF && entity.isAlive()) {
                    ++aliveWolfCount;
                    aliveWolfHealth = ((WolfEntity) entity).getHealth();
                }

                if (semantic > 0 && semantic <= 28 && entity.isAlive()) {
                    latestSceneAliveClassesMask |= 1 << (semantic - 1);
                }
                if (semantic == 0 || !semanticEnabledForGoal(semantic)
                        || entity == minecraft.player) {
                    continue;
                }
                final int entityId = entity.getEntityId();
                if (entityId <= 0 || entityId > MAX_TRANSPORT_ENTITY_ID) {

                    return;
                }
                ids.setId(semantic, entityId);
                renderEntityGeometry(
                    manager, entity, eye, partialTicks, matrix, ids);
                sink.finish();
            }
            latestUniqueWolfHealth = aliveWolfCount == 1
                ? aliveWolfHealth : Float.NaN;

            final int pixels = Math.multiplyExact(width, height);
            if (latestEntityIdRgb == null
                    || latestEntityIdRgb.capacity() != Math.multiplyExact(pixels, 3)) {
                latestEntityIdRgb = BufferUtils.createByteBuffer(
                    Math.multiplyExact(pixels, 3));
            }
            latestEntityIdRgb.clear();
            entityIdFramebuffer.bindFramebuffer(true);
            GL11.glReadPixels(
                0, 0, width, height,
                GL11.GL_RGB, GL11.GL_UNSIGNED_BYTE, latestEntityIdRgb);
            latestEntityIdWidth = width;
            latestEntityIdHeight = height;
            latestEntityIdValid = true;
        } catch (Throwable error) {

            latestEntityIdValid = false;
            if (!entityIdErrorReported) {
                entityIdErrorReported = true;
                System.err.println("entity-ID pass failed; frame marked invalid:");
                error.printStackTrace();
            }
        } finally {
            main.bindFramebuffer(true);
        }
    }

    private static void ensureEntityIdFramebuffer(int width, int height) {
        if (entityIdFramebuffer == null
                || entityIdFramebuffer.framebufferWidth != width
                || entityIdFramebuffer.framebufferHeight != height) {
            if (entityIdFramebuffer != null) {
                entityIdFramebuffer.deleteFramebuffer();
            }
            entityIdFramebuffer = new Framebuffer(
                width, height, true, Minecraft.IS_RUNNING_ON_MAC);
            entityIdFramebuffer.setFramebufferColor(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    private static int semanticCode(Entity entity) {
        final EntityType<?> type = entity.getType();
        if (type == EntityType.SHEEP) {
            final DyeColor color = ((SheepEntity) entity).getFleeceColor();
            if (color == DyeColor.BLACK) {
                return SEMANTIC_BLACK_SHEEP;
            }
            if (color == DyeColor.WHITE) {
                return SEMANTIC_WHITE_SHEEP;
            }
            if (color == DyeColor.GRAY) {
                return SEMANTIC_GRAY_SHEEP;
            }
            if (color == DyeColor.LIGHT_GRAY) {
                return SEMANTIC_LIGHT_GRAY_SHEEP;
            }
            if (color == DyeColor.BROWN) {
                return SEMANTIC_BROWN_SHEEP;
            }
            if (color == DyeColor.BLUE) {
                return SEMANTIC_BLUE_SHEEP;
            }
            if (color == DyeColor.PURPLE) {
                return SEMANTIC_PURPLE_SHEEP;
            }
            if (color == DyeColor.RED) {
                return SEMANTIC_RED_SHEEP;
            }
            if (color == DyeColor.YELLOW) {
                return SEMANTIC_YELLOW_SHEEP;
            }
            if (color == DyeColor.ORANGE) {
                return SEMANTIC_ORANGE_SHEEP;
            }
            return SEMANTIC_SHEEP;
        }
        if (type == EntityType.COW) {
            return SEMANTIC_COW;
        }
        if (type == EntityType.PIG) {
            return SEMANTIC_PIG;
        }
        if (type == EntityType.CHICKEN) {
            return SEMANTIC_CHICKEN;
        }
        if (type == EntityType.HORSE) {
            return SEMANTIC_HORSE;
        }
        if (type == EntityType.PANDA) {
            return SEMANTIC_PANDA;
        }
        if (type == EntityType.MOOSHROOM) {
            return SEMANTIC_MOOSHROOM;
        }
        if (type == EntityType.TURTLE) {
            return SEMANTIC_TURTLE;
        }
        if (type == EntityType.LLAMA) {
            return SEMANTIC_LLAMA;
        }
        if (type == EntityType.DONKEY) {
            return SEMANTIC_DONKEY;
        }
        if (type == EntityType.FOX) {
            return SEMANTIC_FOX;
        }
        if (type == EntityType.CAT) {
            return SEMANTIC_CAT;
        }
        if (type == EntityType.OCELOT) {
            return SEMANTIC_OCELOT;
        }
        if (type == EntityType.POLAR_BEAR) {
            return SEMANTIC_POLAR_BEAR;
        }
        if (type == EntityType.RABBIT) {
            return SEMANTIC_RABBIT;
        }
        if (type == EntityType.PARROT) {
            return SEMANTIC_PARROT;
        }
        if (type == EntityType.WOLF) {
            return SEMANTIC_WOLF;
        }
        if (type == EntityType.TRADER_LLAMA) {
            final int variant = ((LlamaEntity) entity).getVariant();
            if (variant == 0) {
                return SEMANTIC_CREAMY_TRADER_LLAMA;
            }
            if (variant == 1) {
                return SEMANTIC_WHITE_TRADER_LLAMA;
            }
            return 0;
        }
        return 0;
    }

    private static boolean semanticEnabledForGoal(int semantic) {
        final String goal = System.getenv("XBENCH_ENTITY_ID_GOAL");
        if (goal == null || goal.isEmpty()) {
            return true;
        }
        if ("cow".equals(goal)) return semantic == SEMANTIC_COW;
        if ("pig".equals(goal)) return semantic == SEMANTIC_PIG;
        if ("chicken".equals(goal)) return semantic == SEMANTIC_CHICKEN;
        if ("panda".equals(goal)) return semantic == SEMANTIC_PANDA;
        if ("white_sheep".equals(goal)) return semantic == SEMANTIC_WHITE_SHEEP;
        if ("gray_sheep".equals(goal)) return semantic == SEMANTIC_GRAY_SHEEP;
        if ("light_gray_sheep".equals(goal)) {
            return semantic == SEMANTIC_LIGHT_GRAY_SHEEP;
        }
        if ("brown_sheep".equals(goal)) return semantic == SEMANTIC_BROWN_SHEEP;
        if ("blue_sheep".equals(goal)) return semantic == SEMANTIC_BLUE_SHEEP;
        if ("purple_sheep".equals(goal)) return semantic == SEMANTIC_PURPLE_SHEEP;
        if ("creamy_trader_llama".equals(goal)) {
            return semantic == SEMANTIC_CREAMY_TRADER_LLAMA;
        }
        if ("white_trader_llama".equals(goal)) {
            return semantic == SEMANTIC_WHITE_TRADER_LLAMA;
        }
        if ("red_sheep".equals(goal)) return semantic == SEMANTIC_RED_SHEEP;
        if ("yellow_sheep".equals(goal)) return semantic == SEMANTIC_YELLOW_SHEEP;
        if ("orange_sheep".equals(goal)) return semantic == SEMANTIC_ORANGE_SHEEP;
        if ("turtle".equals(goal)) return semantic == SEMANTIC_TURTLE;
        if ("llama".equals(goal)) return semantic == SEMANTIC_LLAMA;
        if ("donkey".equals(goal)) return semantic == SEMANTIC_DONKEY;
        if ("fox".equals(goal)) return semantic == SEMANTIC_FOX;
        if ("cat".equals(goal)) return semantic == SEMANTIC_CAT;
        if ("ocelot".equals(goal)) return semantic == SEMANTIC_OCELOT;
        if ("polar_bear".equals(goal)) return semantic == SEMANTIC_POLAR_BEAR;
        if ("rabbit".equals(goal)) return semantic == SEMANTIC_RABBIT;
        if ("parrot".equals(goal)) return semantic == SEMANTIC_PARROT;
        if ("wolf".equals(goal)) return semantic == SEMANTIC_WOLF;
        return false;
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static void renderEntityGeometry(
            EntityRendererManager manager, Entity entity, Vector3d eye,
            float partialTicks, MatrixStack matrix, IRenderTypeBuffer buffer) {
        final double x = MathHelper.lerp(
            (double) partialTicks, entity.lastTickPosX, entity.getPosX());
        final double y = MathHelper.lerp(
            (double) partialTicks, entity.lastTickPosY, entity.getPosY());
        final double z = MathHelper.lerp(
            (double) partialTicks, entity.lastTickPosZ, entity.getPosZ());
        final float yaw = MathHelper.lerp(
            partialTicks, entity.prevRotationYaw, entity.rotationYaw);
        final EntityRenderer renderer = manager.getRenderer(entity);
        final Vector3d offset = renderer.getRenderOffset(entity, partialTicks);
        matrix.push();
        matrix.translate(
            x - eye.getX() + offset.getX(),
            y - eye.getY() + offset.getY(),
            z - eye.getZ() + offset.getZ());
        try {
            renderer.render(
                entity, yaw, partialTicks, matrix, buffer,
                manager.getPackedLight(entity, partialTicks));
        } finally {
            matrix.pop();
        }
    }

    private static final class EntityIdRenderTypes extends RenderType {
        private static final Map<ResourceLocation, RenderType> CACHE =
            new HashMap<ResourceLocation, RenderType>();
        private static final RenderState.TargetState TARGET =
            new RenderState.TargetState(
                "xbench_entity_id_target",
                () -> {
                    if (entityIdFramebuffer != null) {
                        entityIdFramebuffer.bindFramebuffer(false);
                    }
                },
                () -> Minecraft.getInstance().getFramebuffer()
                    .bindFramebuffer(false));

        private EntityIdRenderTypes() {
            super("xbench_entity_id_base", DefaultVertexFormats.POSITION_COLOR_TEX,
                GL11.GL_QUADS, 256, false, false, () -> {}, () -> {});
        }

        private static RenderType forTexture(ResourceLocation texture) {
            RenderType cached = CACHE.get(texture);
            if (cached != null) {
                return cached;
            }
            final RenderType.State state = RenderType.State.getBuilder()
                .texture(new RenderState.TextureState(texture, false, false))
                .transparency(NO_TRANSPARENCY)
                .alpha(DEFAULT_ALPHA)
                .cull(CULL_DISABLED)
                .depthTest(DEPTH_LEQUAL)
                .lightmap(LIGHTMAP_DISABLED)
                .overlay(OVERLAY_DISABLED)
                .fog(NO_FOG)
                .texturing(OUTLINE_TEXTURING)
                .target(TARGET)
                .writeMask(COLOR_DEPTH_WRITE)
                .build(false);
            cached = RenderType.makeType(
                "xbench_entity_id", DefaultVertexFormats.POSITION_COLOR_TEX,
                GL11.GL_QUADS, 256, false, false, state);
            CACHE.put(texture, cached);
            return cached;
        }
    }

    private static final class EntityIdBuffer implements IRenderTypeBuffer {
        private final IRenderTypeBuffer.Impl sink;
        private int red;
        private int green;
        private int blue;

        private EntityIdBuffer(IRenderTypeBuffer.Impl sink) {
            this.sink = sink;
        }

        private void setId(int semantic, int entityId) {
            red = semantic;
            green = (entityId >> 8) & 0xff;
            blue = entityId & 0xff;
        }

        @Override
        public IVertexBuilder getBuffer(RenderType source) {
            final ResourceLocation texture = source.getTextureLocation();
            if (texture == null) {
                return sink.getDummyBuffer();
            }
            return new EntityIdVertexBuilder(
                sink.getBuffer(EntityIdRenderTypes.forTexture(texture)),
                red, green, blue);
        }
    }

    private static final class EntityIdVertexBuilder
            extends DefaultColorVertexBuilder {
        private final IVertexBuilder sink;
        private double x;
        private double y;
        private double z;
        private float u;
        private float v;

        private EntityIdVertexBuilder(
                IVertexBuilder sink, int red, int green, int blue) {
            this.sink = sink;
            super.setDefaultColor(red, green, blue, 255);
        }

        @Override
        public void setDefaultColor(int red, int green, int blue, int alpha) {

        }

        @Override
        public IVertexBuilder pos(double x, double y, double z) {
            this.x = x;
            this.y = y;
            this.z = z;
            return this;
        }

        @Override
        public IVertexBuilder color(int red, int green, int blue, int alpha) {
            return this;
        }

        @Override
        public IVertexBuilder tex(float u, float v) {
            this.u = u;
            this.v = v;
            return this;
        }

        @Override
        public IVertexBuilder overlay(int u, int v) {
            return this;
        }

        @Override
        public IVertexBuilder lightmap(int u, int v) {
            return this;
        }

        @Override
        public IVertexBuilder normal(float x, float y, float z) {
            return this;
        }

        @Override
        public void endVertex() {
            sink.pos(x, y, z)
                .color(defaultRed, defaultGreen, defaultBlue, 255)
                .tex(u, v)
                .endVertex();
        }
    }

    private static void pollSnapshotSaveRequest(final Minecraft minecraft) {
        if (snapshotSaveInFlight != null || !Files.isRegularFile(SNAPSHOT_SAVE_REQUEST)) {
            return;
        }
        final String nonce;
        try {
            nonce = new String(
                Files.readAllBytes(SNAPSHOT_SAVE_REQUEST),
                StandardCharsets.UTF_8).trim();
        } catch (Exception error) {
            return;
        }
        if (!nonce.matches("[0-9a-f]{32}")) {
            writeSnapshotSaveResult("ERROR", nonce, "invalid_nonce");
            return;
        }
        final MinecraftServer server = minecraft.getIntegratedServer();
        if (server == null) {
            writeSnapshotSaveResult("ERROR", nonce, "no_integrated_server");
            return;
        }
        snapshotSaveInFlight = nonce;
        server.execute(() -> {
            try {
                server.getPlayerList().saveAllPlayerData();

                if (!server.save(false, true, true)) {
                    throw new IllegalStateException("MinecraftServer.save returned false");
                }
                writeSnapshotSaveResult("OK", nonce, "integrated_server_flush_v1");
            } catch (Throwable error) {
                writeSnapshotSaveResult(
                    "ERROR", nonce,
                    error.getClass().getName() + ":" + String.valueOf(error.getMessage()));
            } finally {
                try {
                    Files.deleteIfExists(SNAPSHOT_SAVE_REQUEST);
                } catch (Exception ignored) {

                }
                snapshotSaveInFlight = null;
            }
        });
    }

    private static void writeSnapshotSaveResult(
            String status, String nonce, String detail) {
        final Path temporary = Paths.get(
            SNAPSHOT_SAVE_COMPLETE.toString() + ".tmp");
        try {
            Files.write(
                temporary,
                (status + "\n" + nonce + "\n" + detail + "\n")
                    .getBytes(StandardCharsets.UTF_8));
            try {
                Files.move(
                    temporary, SNAPSHOT_SAVE_COMPLETE,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
            } catch (Exception atomicMoveError) {
                Files.move(
                    temporary, SNAPSHOT_SAVE_COMPLETE,
                    StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (Exception ignored) {

        }
    }

    public static void capture(
            Minecraft minecraft, int width, int height, ByteBuffer output) {

        if ("1".equals(System.getenv("XBENCH_CLEAR_TOASTS"))) {
            minecraft.getToastGui().clear();
        }
        final int pixels = Math.multiplyExact(width, height);
        final boolean withWolfHealth =
            output.capacity() >= Math.multiplyExact(pixels, 20);
        final boolean withSceneClasses = withWolfHealth
            || output.capacity() >= Math.multiplyExact(pixels, 16);
        final boolean withEntityIds = withSceneClasses
            || output.capacity() >= Math.multiplyExact(pixels, 12);
        final boolean withDepth = withEntityIds
            || output.capacity() >= Math.multiplyExact(pixels, 8);
        final boolean withMask = withDepth
            || output.capacity() >= Math.multiplyExact(pixels, 4);

        RenderSystem.pushMatrix();
        minecraft.getFramebuffer().framebufferRender(width, height);
        RenderSystem.popMatrix();
        if (!withMask) {
            GL11.glReadPixels(
                0, 0, width, height,
                GL11.GL_RGB, GL11.GL_UNSIGNED_BYTE, output);
            return;
        }

        final ByteBuffer rgb = BufferUtils.createByteBuffer(
            Math.multiplyExact(pixels, 3));
        GL11.glReadPixels(
            0, 0, width, height,
            GL11.GL_RGB, GL11.GL_UNSIGNED_BYTE, rgb);
        final int stride = withWolfHealth ? 20 : withSceneClasses ? 16 : withEntityIds ? 12
            : withDepth ? 8 : 4;
        final boolean maskReady = latestMaskAlpha != null
            && latestMaskWidth == width && latestMaskHeight == height;
        final boolean depthReady = withDepth && latestDepthFloats != null
            && latestDepthWidth == width && latestDepthHeight == height;
        final boolean entityIdsReady = withEntityIds && latestEntityIdValid
            && latestEntityIdRgb != null
            && latestEntityIdWidth == width && latestEntityIdHeight == height;
        for (int index = 0; index < pixels; ++index) {
            output.put(index * stride, rgb.get(index * 3));
            output.put(index * stride + 1, rgb.get(index * 3 + 1));
            output.put(index * stride + 2, rgb.get(index * 3 + 2));
            final int alpha = maskReady
                ? (latestMaskAlpha.get(index) & 0xff) : -1;

            output.put(index * stride + 3,
                (byte) (alpha > 0 ? 254 : (alpha == 0 ? 255 : 253)));
            if (withDepth) {

                final float value = depthReady
                    ? latestDepthFloats.get(index) : Float.NaN;
                final int bits = Float.floatToRawIntBits(value);
                output.put(index * stride + 4, (byte) (bits & 0xff));
                output.put(index * stride + 5, (byte) ((bits >> 8) & 0xff));
                output.put(index * stride + 6, (byte) ((bits >> 16) & 0xff));
                output.put(index * stride + 7, (byte) ((bits >> 24) & 0xff));
            }
            if (withEntityIds) {
                output.put(index * stride + 8, entityIdsReady
                    ? latestEntityIdRgb.get(index * 3) : (byte) 0);
                output.put(index * stride + 9, entityIdsReady
                    ? latestEntityIdRgb.get(index * 3 + 1) : (byte) 0);
                output.put(index * stride + 10, entityIdsReady
                    ? latestEntityIdRgb.get(index * 3 + 2) : (byte) 0);

                output.put(index * stride + 11,
                    (byte) (entityIdsReady ? 254 : 253));
            }
            if (withSceneClasses) {

                output.put(index * stride + 12,
                    (byte) (latestSceneAliveClassesMask & 0xff));
                output.put(index * stride + 13,
                    (byte) ((latestSceneAliveClassesMask >> 8) & 0xff));
                output.put(index * stride + 14,
                    (byte) ((latestSceneAliveClassesMask >> 16) & 0xff));
                output.put(index * stride + 15,
                    (byte) ((latestSceneAliveClassesMask >> 24) & 0xff));
            }
            if (withWolfHealth) {

                final int bits = Float.floatToRawIntBits(latestUniqueWolfHealth);
                output.put(index * stride + 16, (byte) (bits & 0xff));
                output.put(index * stride + 17, (byte) ((bits >> 8) & 0xff));
                output.put(index * stride + 18, (byte) ((bits >> 16) & 0xff));
                output.put(index * stride + 19, (byte) ((bits >> 24) & 0xff));
            }
        }
    }
}
