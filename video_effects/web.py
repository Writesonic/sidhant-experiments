"""Chainlit web interface for the Video Effects workflow.

Run with: chainlit run video_effects/web.py
"""

import asyncio
import os

import chainlit as cl
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from video_effects.config import settings
from video_effects.schemas.workflow import VideoEffectsInput
from video_effects.web_helpers import format_mg_plan_markdown, format_timeline_markdown


async def _get_client() -> Client:
    return await Client.connect(
        settings.TEMPORAL_ENDPOINT,
        namespace=settings.TEMPORAL_NAMESPACE,
        data_converter=pydantic_data_converter,
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("workflow_id", None)
    cl.user_session.set("temporal_handle", None)
    cl.user_session.set("poll_task", None)
    cl.user_session.set("awaiting_feedback_for", None)

    await cl.Message(
        content=(
            "**VFX Studio**\n\n"
            "Send a message to start a workflow. Format:\n\n"
            "```\n"
            "/path/to/video.mp4 [--programmer] [--mg] [--style <name>] [--dev]\n"
            "```\n\n"
            "I'll guide you through timeline and motion graphics approval."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    awaiting = cl.user_session.get("awaiting_feedback_for")

    # If we're collecting rejection feedback, signal it
    if awaiting:
        handle = cl.user_session.get("temporal_handle")
        feedback = message.content.strip()
        signal_name = "approve_mg" if awaiting == "mg" else "approve_timeline"
        await handle.signal(signal_name, [False, feedback])
        cl.user_session.set("awaiting_feedback_for", None)
        await cl.Message(content=f"Feedback sent. Waiting for revised plan...").send()
        return

    # Otherwise, start a new workflow
    if cl.user_session.get("temporal_handle"):
        await cl.Message(content="A workflow is already running. Wait for it to finish.").send()
        return

    await _start_workflow(message.content.strip())


async def _start_workflow(raw_input: str):
    parts = raw_input.split()
    if not parts:
        await cl.Message(content="Please provide a video path.").send()
        return

    video_path = os.path.abspath(parts[0])
    flags = set(parts[1:])

    enable_programmer = "--programmer" in flags
    enable_mg = "--mg" in flags or enable_programmer
    enable_infographics = "--mg" in flags
    style = ""
    if "--style" in flags:
        idx = parts.index("--style")
        if idx + 1 < len(parts):
            style = parts[idx + 1]
    dev_mode = "--dev" in flags

    base = video_path.rsplit(".", 1)[0]
    output_path = f"{base}_effects.mp4"

    input_data = VideoEffectsInput(
        input_video=video_path,
        output_video=output_path,
        auto_approve=False,
        enable_motion_graphics=enable_mg,
        style=style,
        dev_mode=dev_mode,
        enable_infographics=enable_infographics,
        enable_programmer=enable_programmer,
    )

    import uuid
    workflow_id = f"vfx-web-{uuid.uuid4().hex[:8]}"

    client = await _get_client()
    handle = await client.start_workflow(
        "VideoEffectsWorkflow",
        input_data,
        id=workflow_id,
        task_queue=settings.TASK_QUEUE,
    )

    cl.user_session.set("workflow_id", workflow_id)
    cl.user_session.set("temporal_handle", handle)

    await cl.Message(content=f"Workflow **{workflow_id}** started.\n\nInput: `{video_path}`").send()

    poll_task = asyncio.create_task(_poll_workflow(handle))
    cl.user_session.set("poll_task", poll_task)


# ---------------------------------------------------------------------------
# Action callbacks
# ---------------------------------------------------------------------------


@cl.action_callback("approve_timeline")
async def on_approve_timeline(action: cl.Action):
    handle = cl.user_session.get("temporal_handle")
    if handle:
        await handle.signal("approve_timeline", [True, ""])
        await cl.Message(content="Timeline approved. Processing...").send()


@cl.action_callback("reject_timeline")
async def on_reject_timeline(action: cl.Action):
    cl.user_session.set("awaiting_feedback_for", "timeline")
    await cl.Message(content="What would you like changed? Type your feedback below.").send()


@cl.action_callback("approve_mg")
async def on_approve_mg(action: cl.Action):
    handle = cl.user_session.get("temporal_handle")
    if handle:
        await handle.signal("approve_mg", [True, ""])
        await cl.Message(content="MG plan approved. Rendering final video...").send()


@cl.action_callback("reject_mg")
async def on_reject_mg(action: cl.Action):
    cl.user_session.set("awaiting_feedback_for", "mg")
    await cl.Message(content="What would you like changed? Type your feedback below.").send()


# ---------------------------------------------------------------------------
# Background poller
# ---------------------------------------------------------------------------


async def _poll_workflow(handle):
    prev_stage = None

    while True:
        try:
            stage = await handle.query("get_workflow_stage")
        except Exception:
            await asyncio.sleep(2)
            continue

        if stage == prev_stage:
            await asyncio.sleep(2)
            continue
        prev_stage = stage

        if stage == "analyzing":
            await cl.Message(content="Analyzing video...").send()

        elif stage == "timeline_approval":
            timeline = await handle.query("get_timeline")
            md = format_timeline_markdown(timeline) if timeline else "No timeline data."

            actions = [
                cl.Action(name="approve_timeline", label="Approve", payload={}),
                cl.Action(name="reject_timeline", label="Reject", payload={}),
            ]
            await cl.Message(
                content=f"## Effects Timeline\n\n{md}\n\nReview and approve or reject.",
                actions=actions,
            ).send()

        elif stage == "processing":
            await cl.Message(content="Timeline approved. Processing effects and generating components...").send()

        elif stage == "mg_preview":
            await cl.Message(content="Rendering MG preview snapshots...").send()

        elif stage == "mg_approval":
            plan = await handle.query("get_mg_plan")
            preview = await handle.query("get_mg_preview")

            md = format_mg_plan_markdown(plan) if plan else "No MG plan."

            elements = []
            if preview and preview.get("snapshots"):
                for snap in preview["snapshots"]:
                    path = snap.get("path", "")
                    if path and os.path.exists(path):
                        elements.append(
                            cl.Image(path=path, name=f"Frame {snap['frame']}", display="inline", size="large")
                        )

            actions = [
                cl.Action(name="approve_mg", label="Approve", payload={}),
                cl.Action(name="reject_mg", label="Reject", payload={}),
            ]
            await cl.Message(
                content=f"## Motion Graphics Plan\n\n{md}\n\nReview the preview and approve or reject.",
                elements=elements,
                actions=actions,
            ).send()

        elif stage == "rendering":
            await cl.Message(content="Rendering final video...").send()

        elif stage == "done":
            try:
                result = await handle.result()
                if isinstance(result, dict):
                    output = result.get("output_video", "N/A")
                    effects = result.get("effects_applied", 0)
                    mg = result.get("motion_graphics_applied", 0)
                else:
                    output = result.output_video
                    effects = result.effects_applied
                    mg = result.motion_graphics_applied

                summary = f"**Done!**\n\n- Output: `{output}`\n- Effects: {effects}"
                if mg:
                    summary += f"\n- Motion graphics: {mg} components"
                await cl.Message(content=summary).send()
            except Exception as e:
                await cl.Message(content=f"Workflow finished with error: {e}").send()
            break

        elif stage == "error":
            try:
                result = await handle.result()
                error = result.get("error") if isinstance(result, dict) else result.error
                await cl.Message(content=f"Workflow error: {error}").send()
            except Exception as e:
                await cl.Message(content=f"Workflow error: {e}").send()
            break

        await asyncio.sleep(2)

    cl.user_session.set("temporal_handle", None)
    cl.user_session.set("poll_task", None)
